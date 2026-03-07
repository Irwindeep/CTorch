#ifndef VISION_H
#define VISION_H

#include <stdbool.h>
#include <stddef.h>

typedef struct Image Image;

void VisionInit(void);
void VisionClose(void);

Image *image_init(void *image);
Image *load_image(const char *path);
void save_image(Image *image, const char *path);
void free_image(Image *image);

#endif // !VISION_H
