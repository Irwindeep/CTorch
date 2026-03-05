#include "vision/vision.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <vips/vips.h>

struct Image {
    VipsImage *base;
};

void VisionInit(void) { VIPS_INIT("ctorchvision"); }
void VisionClose(void) { vips_shutdown(); }

Image *load_image(const char *path) {
    VipsImage *image = vips_image_new_from_file(path, NULL);
    if (!image)
        return NULL;

    Image *output = malloc(sizeof(Image));
    output->base = image;

    return output;
}

void free_image(Image *image) {
    g_object_unref(image->base);
    free(image);
}
