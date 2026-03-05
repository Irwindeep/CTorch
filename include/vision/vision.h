#ifndef VISION_H
#define VISION_H

#include <stdbool.h>
#include <stddef.h>
#include <vips/vips.h>

void VisionInit(void);
void VisionClose(void);

VipsImage *load_image(const char *path);
void free_image(VipsImage *image);

#endif // !VISION_H
