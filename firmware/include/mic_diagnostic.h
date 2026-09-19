#pragma once
#include <driver/i2s.h>
#include <cmath>
#include "config.h"

namespace micTest {
struct Stats {
    uint32_t count = 0, nonzero = 0, pcmNonzero = 0, peak = 0, last = 0;
    double sum = 0, energy = 0;
    void add(int32_t raw) {
        ++count;
        if (raw != 0) ++nonzero;
        if ((raw >> 16) != 0) ++pcmNonzero;
        uint32_t magnitude = raw < 0 ? uint32_t(-int64_t(raw)) : uint32_t(raw);
        if (magnitude > peak) peak = magnitude;
        last = uint32_t(raw);
        double x = raw / 2147483648.0;
        sum += x;
        energy += x*x;
    }
    void print(int slot) const {
        double mean = count ? sum/count : 0;
        double power = count ? energy/count : 0;
        Serial.printf("slot%d: amostras=%lu raw_nao_zero=%lu pcm16_nao_zero=%lu pico_raw=%lu rms=%.3e ac_rms=%.3e ultimo=0x%08lx\n",
            slot, (unsigned long)count, (unsigned long)nonzero, (unsigned long)pcmNonzero,
            (unsigned long)peak, sqrt(power), sqrt(fmax(0.0, power-mean*mean)), (unsigned long)last);
    }
};
static QueueHandle_t events;
static Stats stats[2];
static uint32_t errors = 0, drops = 0, printedAt = 0;
}

void setup() {
    Serial.begin(115200);
    pinMode(cfg::led, OUTPUT);
    digitalWrite(cfg::led, LOW);
    Serial.println("MIC_TEST: captura bruta dos DOIS canais; sem modelo de reconhecimento.");
    Serial.printf("SCK=%d WS=%d SD=%d | L/R=GND | VDD=3V3 | 16000 Hz, 32 bits por canal\n",
                  cfg::bclk, cfg::ws, cfg::din);
    Serial.println("Slot0/slot1 sao as duas posicoes no buffer; nao presumimos a ordem L/R.");
    i2s_config_t config = {};
    config.mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX);
    config.sample_rate = cfg::sampleRate;
    config.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
    config.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT;
    config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
    config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
    config.dma_buf_count = 8;
    config.dma_buf_len = 128;
    i2s_pin_config_t pins = {};
    pins.mck_io_num = I2S_PIN_NO_CHANGE;
    pins.bck_io_num = cfg::bclk;
    pins.ws_io_num = cfg::ws;
    pins.data_out_num = I2S_PIN_NO_CHANGE;
    pins.data_in_num = cfg::din;
    ESP_ERROR_CHECK(i2s_driver_install(I2S_NUM_0, &config, 16, &micTest::events));
    ESP_ERROR_CHECK(i2s_set_pin(I2S_NUM_0, &pins));
    micTest::printedAt = millis();
}

void loop() {
    static int32_t raw[256];
    size_t bytes = 0;
    esp_err_t result = i2s_read(I2S_NUM_0, raw, sizeof(raw), &bytes, pdMS_TO_TICKS(200));
    if (result != ESP_OK || bytes == 0 || bytes % 8 != 0) {
        ++micTest::errors;
    } else {
        for (size_t i = 0; i < bytes / sizeof(raw[0]); i += 2) {
            micTest::stats[0].add(raw[i]);
            micTest::stats[1].add(raw[i+1]);
        }
    }
    i2s_event_t event;
    while (xQueueReceive(micTest::events, &event, 0) == pdPASS) {
        if (event.type == I2S_EVENT_RX_Q_OVF || event.type == I2S_EVENT_DMA_ERROR) ++micTest::drops;
    }
    uint32_t now = millis();
    if (now - micTest::printedAt >= 1000) {
        Serial.printf("MIC_TEST intervalo_ms=%lu erros=%lu perdas=%lu\n",
            (unsigned long)(now-micTest::printedAt), (unsigned long)micTest::errors, (unsigned long)micTest::drops);
        micTest::stats[0].print(0);
        micTest::stats[1].print(1);
        micTest::stats[0] = micTest::Stats();
        micTest::stats[1] = micTest::Stats();
        micTest::printedAt = now;
    }
}
