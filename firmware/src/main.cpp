#include <Arduino.h>
#include <esp_timer.h>
#include <esp_heap_caps.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <freertos/semphr.h>
#include <freertos/task.h>
#include "config.h"

#ifdef MIC_TEST
#include "mic_diagnostic.h"
#else
#include "audio_input.h"

#ifdef CAPTURE_ONLY
// Firmware dedicado a coletar o dataset com o mesmo ADC e ganho da inferência.
void setup() { Serial.begin(921600); initAudio(); }
void loop() {
    static int16_t pcm[cfg::chunk];
    if (Serial.available() && Serial.read() == 'R') {
        // Drena DMA antes de sinalizar início; erros antigos não contaminam o clipe.
        for (int i = 0; i < 8; ++i) readChunk(pcm);
        Serial.print("PCM67\n");
        for (int n = 0; n < cfg::samples; n += cfg::chunk) {
            if (!readChunk(pcm)) return; // Cliente acusa frame incompleto.
            Serial.write(reinterpret_cast<uint8_t*>(pcm), sizeof(pcm));
        }
        Serial.flush();
    } else {
        readChunk(pcm); // Drena continuamente o DMA enquanto espera um comando.
    }
}
#else
#include "dsp.h"
#include "inference.h"

struct FeaturePacket {
    float values[cfg::features];
    float rms, centroid;
    int64_t endUs, capturedUs, enqueuedUs;
    uint32_t epoch;
    uint32_t sequence, readUs, copyUs, featureUs, skipped, queueDrops, readErrors;
#ifdef SIMULATED_AUDIO
    int clip;
    bool paused;
#endif
};
// Alocados uma vez no heap interno: o ESP32 tem limite menor para DRAM estática.
static int16_t* ring;
static int16_t* snapshot;
static size_t writeIndex = 0;
static uint32_t filled = 0, sinceWindow = 0, sequence = 0;
static uint32_t captureReadUs = 0, captureCopyUs = 0, readErrors = 0;
static int64_t windowEndUs = 0;
static int64_t capturedUs = 0;
static uint32_t captureEpoch = 0;
#ifdef SIMULATED_AUDIO
static int capturedClip = 0;
static bool capturedPaused = false;
#endif
static SemaphoreHandle_t ringMutex;
static QueueHandle_t featuresQueue;
static TaskHandle_t featureHandle;

static void captureTask(void*) {
    int16_t pcm[cfg::chunk];
    while (true) {
        int64_t started = esp_timer_get_time();
        bool ok = readChunk(pcm); // Bloqueia no driver/DMA, sem busy-wait.
        int64_t readEnd = esp_timer_get_time();
        xSemaphoreTake(ringMutex, portMAX_DELAY);
        if (sourceRestarted) { filled = sinceWindow = 0; ++captureEpoch; }
#ifdef SIMULATED_AUDIO
        capturedClip = simPlayer.clip();
        capturedPaused = simPlayer.paused();
#endif
        if (!ok) {
            ++readErrors;
            ++captureEpoch;
            filled = sinceWindow = 0; // Nunca unir áudio separado por falha de captura.
            xSemaphoreGive(ringMutex);
            continue;
        }
        for (int i = 0; i < cfg::chunk; ++i) {
            ring[writeIndex] = pcm[i];
            writeIndex = (writeIndex + 1) % cfg::samples;
        }
        filled = min(static_cast<uint32_t>(cfg::samples), filled + cfg::chunk);
        sinceWindow += cfg::chunk;
        captureReadUs = readEnd-started;
        captureCopyUs = esp_timer_get_time()-readEnd;
        windowEndUs = readEnd; // Timestamp corresponde ao snapshot realmente copiado.
        capturedUs = readEnd;
#ifdef SERIAL_MIC
        windowEndUs = micSamples * 1000000ULL / cfg::sampleRate;
#endif
        bool ready = filled == cfg::samples && sinceWindow >= cfg::stride;
        if (ready) {
            sinceWindow = 0;
            ++sequence;
        }
        xSemaphoreGive(ringMutex);
        if (ready) xTaskNotifyGive(featureHandle);
    }
}

static void featureTask(void*) {
    static Dsp dsp;
    static FeaturePacket packet;
    uint32_t previous = 0, skipped = 0, queueDrops = 0;
    while (true) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        xSemaphoreTake(ringMutex, portMAX_DELAY);
        if (filled != cfg::samples || sequence == previous) {
            xSemaphoreGive(ringMutex);
            continue;
        }
        // Cópia breve sob mutex com herança de prioridade. FFT fora do mutex.
        size_t tail = cfg::samples-writeIndex;
        memcpy(snapshot, ring+writeIndex, tail*sizeof(int16_t));
        memcpy(snapshot+tail, ring, writeIndex*sizeof(int16_t));
        packet.endUs = windowEndUs;
        packet.capturedUs = capturedUs;
        packet.epoch = captureEpoch;
        packet.sequence = sequence;
        packet.readUs = captureReadUs;
        packet.copyUs = captureCopyUs;
        packet.readErrors = readErrors;
#ifdef SIMULATED_AUDIO
        packet.clip = capturedClip;
        packet.paused = capturedPaused;
#endif
        xSemaphoreGive(ringMutex);
        skipped += packet.sequence-previous-1;
        previous = packet.sequence;
        int64_t started = esp_timer_get_time();
        dsp.extract(snapshot, packet.values, packet.rms, packet.centroid);
        packet.featureUs = esp_timer_get_time()-started;
        packet.skipped = skipped;
        packet.queueDrops = queueDrops;
        packet.enqueuedUs = esp_timer_get_time();
        // Política explícita: manter trabalho recente, nunca bloquear captura.
        if (xQueueSend(featuresQueue, &packet, 0) != pdPASS) {
            static FeaturePacket discarded;
            if (xQueueReceive(featuresQueue, &discarded, 0) == pdPASS) ++queueDrops;
            packet.queueDrops = queueDrops;
            packet.enqueuedUs = esp_timer_get_time();
            configASSERT(xQueueSend(featuresQueue, &packet, 0) == pdPASS);
        }
        vTaskDelay(1); // Garante oportunidade à detecção/idle sob sobrecarga contínua.
    }
}

static void setAlert(bool on) {
    digitalWrite(cfg::led, on ? HIGH : LOW);
    if (cfg::buzzer >= 0) digitalWrite(cfg::buzzer, on ? HIGH : LOW);
}

static void detectionTask(void*) {
    static FeaturePacket packet;
    EventGate gate;
    uint32_t lastEpoch = 0;
    int64_t alertUntil = 0;
#ifdef SIMULATED_AUDIO
    int lastClip = -1;
    bool lastPaused = false;
#endif
    while (true) {
        if (esp_timer_get_time() >= alertUntil) setAlert(false);
        if (xQueueReceive(featuresQueue, &packet, pdMS_TO_TICKS(20)) != pdPASS) continue;
        int64_t start = esp_timer_get_time();
        float score = predict(packet.values);
        uint32_t inferenceUs = esp_timer_get_time()-start;
        if (packet.epoch != lastEpoch) { gate = EventGate(); lastEpoch = packet.epoch; }
        bool event = gate.update(score, packet.endUs);
        if (event) {
            setAlert(true);
            alertUntil = esp_timer_get_time()+cfg::alertUs;
        }
        uint32_t pipelineUs = esp_timer_get_time()-packet.capturedUs;
#ifdef SIMULATED_AUDIO
        if (lastClip != packet.clip || lastPaused != packet.paused) {
            Serial.printf("SIM_CLIP id=%d name=%s paused=%d\n", packet.clip, sim::clipName(packet.clip), packet.paused);
            lastClip = packet.clip; lastPaused = packet.paused;
        }
        if (event) {
#ifdef SIMULATOR_DEMO
            Serial.println("DEMO_EVENT: par de tons sinteticos; NAO e reconhecimento de fala");
#else
            Serial.println("AUDIO_EVENT: modelo treinado sinalizou o alvo no WAV");
#endif
        }
#endif
        Serial.printf("{\"seq\":%lu,\"window_end_us\":%lld,\"score\":%.5f,\"event\":%s,\"rms\":%.6f,\"centroid_hz\":%.1f,"
            "\"capture_read_us\":%lu,\"capture_copy_us\":%lu,\"features_us\":%lu,\"queue_us\":%lu,"
            "\"inference_us\":%lu,\"pipeline_us\":%lu,\"skipped\":%lu,\"queue_drops\":%lu,"
            "\"read_errors\":%lu,\"heap\":%u}\n",
            (unsigned long)packet.sequence, (long long)packet.endUs, score, event ? "true" : "false", packet.rms, packet.centroid,
            (unsigned long)packet.readUs, (unsigned long)packet.copyUs, (unsigned long)packet.featureUs,
            (unsigned long)(start-packet.enqueuedUs), (unsigned long)inferenceUs, (unsigned long)pipelineUs,
            (unsigned long)packet.skipped, (unsigned long)packet.queueDrops,
            (unsigned long)packet.readErrors, ESP.getFreeHeap());
    }
}

void setup() {
#ifdef SERIAL_MIC
    Serial.setRxBufferSize(4096);
    Serial.begin(921600);
#else
    Serial.begin(115200);
#endif
    pinMode(cfg::led, OUTPUT);
    if (cfg::buzzer >= 0) pinMode(cfg::buzzer, OUTPUT);
    setAlert(false);
    Serial.printf("Detector 6 7 | model_ready=%d | threshold=%.5f\n", MODEL_READY, MODEL_THRESHOLD);
    Serial.printf("MODEL_ID=%s\n", MODEL_ID);
    Serial.println("Audio: janela de 2 segundos, atualizada a cada 250 ms (4 analises/s).");
#ifdef SIMULATOR_DEMO
    Serial.println("SIM_READY mode=demo | TONS SINTETICOS | SEM MODELO DE FALA");
#elif defined(SERIAL_MIC)
    Serial.println("SIM_READY mode=mic | MICROFONE DO PC VIA SERIAL | MODELO NO ESP32");
#elif defined(SIMULATOR_REAL)
    Serial.println("SIM_READY mode=audio | PCM EM FLASH + MODELO TREINADO | SEM MICROFONE I2S");
#endif
#if !defined(SERIAL_MIC) && !defined(SIMULATED_AUDIO)
    Serial.printf("I2S: stereo -> slot0 | SCK=%d WS=%d SD=%d | L/R=GND\n", cfg::bclk, cfg::ws, cfg::din);
#endif
#ifdef SIMULATED_AUDIO
    Serial.println("Controles: n=proximo, r=reiniciar, p=pausar/retomar. Playlist automatica.");
#endif
    ring = static_cast<int16_t*>(heap_caps_malloc(cfg::samples*sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    snapshot = static_cast<int16_t*>(heap_caps_malloc(cfg::samples*sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    configASSERT(ring && snapshot);
    ringMutex = xSemaphoreCreateMutex();
    featuresQueue = xQueueCreate(2, sizeof(FeaturePacket));
    configASSERT(ringMutex && featuresQueue);
    initAudio();
    configASSERT(xTaskCreatePinnedToCore(featureTask, "features", 8192, nullptr, 3, &featureHandle, 1) == pdPASS);
    configASSERT(xTaskCreatePinnedToCore(detectionTask, "detection", 6144, nullptr, 2, nullptr, 1) == pdPASS);
    configASSERT(xTaskCreatePinnedToCore(captureTask, "capture", 4096, nullptr, 5, nullptr, 0) == pdPASS);
}
void loop() {
#ifdef SIMULATED_AUDIO
    pollSimulationControls();
    vTaskDelay(pdMS_TO_TICKS(10));
#else
    vTaskDelay(pdMS_TO_TICKS(1000));
#endif
}
#endif
#endif // MIC_TEST
