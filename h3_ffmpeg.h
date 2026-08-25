#ifndef H3_FFMPEG_H
#define H3_FFMPEG_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    H3_IMAGE_FIT_STRETCH = 0,
    H3_IMAGE_FIT_COVER = 1
} h3_image_fit;

/* Inspect the first visual stream without decoding it. H3_FFPROBE may select
 * an explicit ffprobe-compatible executable. */
int h3_ffprobe_visual_size(const char *path, int *width, int *height,
                           char *error, size_t error_size);

/* Decode one visual stream through FFmpeg. The caller owns channel-major F32
 * [3,height,width] RGB in [0,1]. */
int h3_ffmpeg_read_image_f32(const char *path, int width, int height,
                             h3_image_fit fit, float **pixels,
                             char *error, size_t error_size);

/* Decode a 24 fps visual stream to channel-major F32 [3,T,H,W] in [0,1].
 * The returned frame count is trimmed down to the released 5+17k cadence. */
int h3_ffmpeg_read_video_f32(const char *path, int width, int height,
                             int max_frames, float **pixels, int *frames,
                             char *error, size_t error_size);

/* Decode the first audio stream as channel-major stereo F32 at 32 kHz.
 * max_samples bounds allocation. truncate_at_limit is used for a video's
 * soundtrack; standalone clips report an error instead of silently trimming. */
int h3_ffmpeg_read_audio_f32(const char *path, int max_samples,
                             int truncate_at_limit,
                             float **pcm, int *samples,
                             char *error, size_t error_size);

int h3_ffmpeg_write_rgb24(const char *path, const uint8_t *frames,
                          int frame_count, int width, int height, int fps,
                          char *error, size_t error_size);

/* Encode RGB24 video and channel-major F32 PCM through two concurrent pipes.
 * No intermediate uncompressed media file is created. */
int h3_ffmpeg_write_av_rgb24_f32(const char *path, const uint8_t *frames,
                                 int frame_count, int width, int height,
                                 int fps, const float *pcm, int samples,
                                 int channels, int sample_rate,
                                 char *error, size_t error_size);

typedef struct h3_ffmpeg_writer h3_ffmpeg_writer;

/* Incremental counterpart to h3_ffmpeg_write_av_rgb24_f32: the whole PCM
 * track is already known up front (audio decodes before video in the
 * generation pipeline) and starts streaming to FFmpeg immediately, but
 * video frames are handed over in chunks via h3_ffmpeg_writer_write_video
 * as they're produced instead of requiring the whole video to already be in
 * memory. FFmpeg encodes each chunk as it arrives, on its own process, so
 * that work overlaps with decoding/upscaling the next chunk. */
h3_ffmpeg_writer *h3_ffmpeg_writer_open(const char *path, int width,
                                        int height, int fps,
                                        const float *pcm, int samples,
                                        int channels, int sample_rate,
                                        char *error, size_t error_size);
int h3_ffmpeg_writer_write_video(h3_ffmpeg_writer *writer,
                                 const uint8_t *frames, int frame_count,
                                 char *error, size_t error_size);
/* Ends the video stream, waits for FFmpeg to finish, and frees the writer
 * either way - always call this exactly once for a writer that was opened,
 * even after a write_video failure, to reap the FFmpeg process. */
int h3_ffmpeg_writer_close(h3_ffmpeg_writer *writer,
                           char *error, size_t error_size);

#endif
