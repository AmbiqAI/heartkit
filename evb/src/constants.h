#ifndef __CONSTANTS_H
#define __CONSTANTS_H

#define SAMPLE_RATE (250)
#define MAX86150_ADDR (0x5E)
#define NUM_ELEMENTS (1)
#define INF_WINDOW_LEN (SAMPLE_RATE*5) // 5 seconds
#define PAD_WINDOW_LEN (SAMPLE_RATE*1) // 1 seconds
#define SENSOR_BUFFER_LEN (PAD_WINDOW_LEN+INF_WINDOW_LEN+SAMPLE_RATE)

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

static const char *heart_rhythm_labels[] = { "normal", "afib", "aflut", "noise" };
// static const int SENSOR_ELEMENTS[] = [9, 0, 0, 0];
// static const char *heart_beat_labels[] = { "normal", "pac", "aberrated", "pvc", "noise" };
// static const char *hear_rate_labels[] = { "normal", "tachycardia", "bradycardia", "noise" };

#endif // __CONSTANTS_H
