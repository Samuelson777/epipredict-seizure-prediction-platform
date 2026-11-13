/**
 * edge_dsp_daemon.cpp
 * High-Performance C++17 Edge Acquisition, Lock-Free Ring Buffering,
 * and BrainFlow Multicast Telemetry Streamer for EpiPredict.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <numeric>

// BrainFlow C++ SDK Headers (When compiled with -DUSE_BRAINFLOW)
#ifdef USE_BRAINFLOW
#include "board_shim.h"
#include "data_filter.h"
#endif

class LockFreeRingBuffer {
public:
    explicit LockFreeRingBuffer(size_t capacity)
        : capacity_(capacity), buffer_(capacity, 0.0f), head_(0), size_(0) {}

    void push(float sample) noexcept {
        buffer_[head_] = sample;
        head_ = (head_ + 1) % capacity_;
        if (size_ < capacity_) {
            ++size_;
        }
    }

    double computeRMS() const noexcept {
        if (size_ == 0) return 0.0;
        double sum_sq = 0.0;
        for (size_t i = 0; i < size_; ++i) {
            sum_sq += static_cast<double>(buffer_[i]) * static_cast<double>(buffer_[i]);
        }
        return std::sqrt(sum_sq / static_cast<double>(size_));
    }

private:
    size_t capacity_;
    std::vector<float> buffer_;
    size_t head_;
    size_t size_;
};

int main(int argc, char* argv[]) {
    std::cout << "[EpiPredict C++ Edge Daemon] Initializing Low-Latency Biopotential Engine...\n";

    constexpr int kSamplingRateHz = 256;
    constexpr int kWindowSeconds = 5;
    constexpr size_t kBufferCapacity = kSamplingRateHz * kWindowSeconds;

    LockFreeRingBuffer eeg_ch0_buffer(kBufferCapacity);

#ifdef USE_BRAINFLOW
    BoardShim::enable_dev_board_logger();
    BrainFlowInputParams params;
    int board_id = static_cast<int>(BoardIds::SYNTHETIC_BOARD); // Switch to MUSE_S_BOARD or GANGLION_NATIVE_BOARD

    BoardShim board(board_id, params);
    try {
        board.prepare_session();
        board.add_streamer("streaming_board://225.1.1.1:6677", static_cast<int>(BrainFlowPresets::DEFAULT_PRESET));
        board.start_stream(45000);
        std::cout << "[BrainFlow C++] Multicast Stream Active on streaming_board://225.1.1.1:6677\n";

        std::this_thread::sleep_for(std::chrono::seconds(2));
        BrainFlowArray<double, 2> data = board.get_current_board_data(256, static_cast<int>(BrainFlowPresets::DEFAULT_PRESET));
        int eeg_Len = 0;
        int* eeg_channels = BoardShim::get_eeg_channels(board_id, &eeg_Len);

        for (int i = 0; i < eeg_Len; ++i) {
            int ch = eeg_channels[i];
            DataFilter::perform_bandpass(data.get_address(ch), data.get_size(1), kSamplingRateHz, 0.5, 45.0, 4,
                                         static_cast<int>(FilterTypes::BUTTERWORTH_ZERO_PHASE), 0.0);
            DataFilter::remove_environmental_noise(data.get_address(ch), data.get_size(1), kSamplingRateHz,
                                                   static_cast<int>(NoiseTypes::FIFTY_AND_SIXTY));
        }
        delete[] eeg_channels;
        board.stop_stream();
        board.release_session();
    } catch (const BrainFlowException& err) {
        std::cerr << "[BrainFlow Exception] " << err.what() << "\n";
    }
#else
    // Standalone deterministic simulation benchmark when BrainFlow shared libs are not linked
    auto t_start = std::chrono::high_resolution_clock::now();
    for (size_t n = 0; n < kBufferCapacity; ++n) {
        double t = static_cast<double>(n) / kSamplingRateHz;
        float synthetic_uv = static_cast<float>(25.0 * std::sin(2.0 * M_PI * 10.0 * t) +
                                                8.5 * std::sin(2.0 * M_PI * 42.0 * t));
        eeg_ch0_buffer.push(synthetic_uv);
    }
    double rms_uv = eeg_ch0_buffer.computeRMS();
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

    std::cout << "[EpiPredict C++ Benchmark] Processed " << kBufferCapacity
              << " samples in " << elapsed_us << " us | RMS Amplitude = " << rms_uv << " uV\n";
#endif

    return 0;
}