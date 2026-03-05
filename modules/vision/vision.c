#include "vision/vision.h"

#include <stdbool.h>
#include <vips/vips.h>

void VisionInit(void) { VIPS_INIT("ctorchvision"); }
void VisionClose(void) { vips_shutdown(); }

VipsImage *load_image(const char *path) {
    VipsImage *image = vips_image_new_from_file(path, NULL);
    if (!image)
        return NULL;

    return image;
}

void free_image(VipsImage *image) { g_object_unref(image); }
