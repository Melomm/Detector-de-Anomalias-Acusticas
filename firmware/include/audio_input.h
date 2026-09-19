#pragma once
#include <Arduino.h>
#include "config.h"

static bool sourceRestarted = false;
#ifdef SERIAL_MIC
#include "mic_protocol.h"
static mic::Parser micParser;
static uint64_t micSamples = 0;
static void initAudio() {}
static bool readChunk(int16_t* pcm) {
    static size_t offset = mic::samples;
    static uint32_t previous = 0, lastByteAt = 0;
    static bool first = true;
    sourceRestarted = false;
    if (offset == mic::samples) {
        while (true) {
            if (!Serial.available()) {
                if (micParser.partial() && millis() - lastByteAt > 2000) {
                    micParser.reset();
                    Serial.println("MIC67_ERROR incomplete");
                    return false;
                }
                vTaskDelay(1);
                continue;
            }
            lastByteAt = millis();
            int result = micParser.feed(static_cast<uint8_t>(Serial.read()));
            if (result < 0) {
                Serial.println("MIC67_ERROR crc");
                return false;
            }
            if (result == 1) break;
        }
        sourceRestarted = first || micParser.sequence() != previous + 1;
        first = false;
        previous = micParser.sequence();
        offset = 0;
    }
    micParser.copy(offset, pcm, cfg::chunk);
    offset += cfg::chunk;
    micSamples += cfg::chunk;
    // Yield to DSP/idle; timing of the audio comes from samples, not network delay.
    vTaskDelay(1);
    if (offset == mic::samples) Serial.printf("MIC67_ACK %lu\n", (unsigned long)previous);
    return true;
}
#elif defined(SIMULATED_AUDIO)
#include "simulation_player.h"
static sim::Player simPlayer;
static QueueHandle_t simCommands;
static const int simButtonPins[] = {18, 19, 23};

static void initAudio() {
    simCommands = xQueueCreate(8, sizeof(sim::Command));
    configASSERT(simCommands);
    for (int pin : simButtonPins) pinMode(pin, INPUT_PULLUP);
}

static void pollSimulationControls() {
    const sim::Command commands[] = {sim::Command::Next, sim::Command::Restart, sim::Command::Pause};
    static bool previous[] = {false, false, false};
    static uint32_t pressedAt[] = {0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        bool pressed = digitalRead(simButtonPins[i]) == LOW;
        if (pressed && !previous[i] && millis()-pressedAt[i] >= 150) {
            xQueueSend(simCommands, &commands[i], 0);
            pressedAt[i] = millis();
        }
        previous[i] = pressed;
    }
    while (Serial.available()) {
        char c = Serial.read();
        int index = c == 'n' ? 0 : c == 'r' ? 1 : c == 'p' ? 2 : -1;
        if (index >= 0) xQueueSend(simCommands, &commands[index], 0);
    }
}

static bool readChunk(int16_t* pcm) {
    static TickType_t wake = xTaskGetTickCount();
    // 160 amostras a cada 10 ms de tempo SIMULADO, independentemente do PC.
    vTaskDelayUntil(&wake, pdMS_TO_TICKS(10));
    sim::Command command;
    while (xQueueReceive(simCommands, &command, 0) == pdPASS) simPlayer.command(command);
    sourceRestarted = simPlayer.read(pcm);
    return true;
}
#else
#include <driver/i2s.h>
static QueueHandle_t i2sEvents;

static void initAudio() {
    i2s_config_t config = {};
    config.mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX);
    config.sample_rate = cfg::sampleRate;
    config.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
    // Stereo matches MIC_TEST. This board's valid microphone samples are in slot0.
    config.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT;
    config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
    config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
    config.dma_buf_count = 8;
    config.dma_buf_len = cfg::chunk;
    i2s_pin_config_t pins = {};
    pins.mck_io_num = I2S_PIN_NO_CHANGE;
    pins.bck_io_num = cfg::bclk;
    pins.ws_io_num = cfg::ws;
    pins.data_out_num = I2S_PIN_NO_CHANGE;
    pins.data_in_num = cfg::din;
    ESP_ERROR_CHECK(i2s_driver_install(I2S_NUM_0, &config, 16, &i2sEvents));
    ESP_ERROR_CHECK(i2s_set_pin(I2S_NUM_0, &pins));
}

static bool readChunk(int16_t* pcm) {
    int32_t raw[cfg::chunk * 2];
    size_t count = 0;
    esp_err_t err = i2s_read(I2S_NUM_0, raw, sizeof(raw), &count, pdMS_TO_TICKS(100));
    if (err != ESP_OK || count != sizeof(raw)) return false;
    i2s_event_t event;
    bool continuous = true;
    while (xQueueReceive(i2sEvents, &event, 0) == pdPASS) {
        if (event.type == I2S_EVENT_RX_Q_OVF || event.type == I2S_EVENT_DMA_ERROR) continuous = false;
    }
    if (!continuous) return false;
    // One mono sample per stereo frame: preserve 16 kHz and the existing gain.
    for (int i = 0; i < cfg::chunk; ++i) pcm[i] = static_cast<int16_t>(raw[2 * i] >> 16);
    return true;
}
#endif
