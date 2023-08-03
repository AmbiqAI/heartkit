// Handle TinyUSB events

#include "usb_handler.h"

bool volatile *g_usb_available;

void
tud_mount_cb(void) {
    *g_usb_available = true;
}
void
tud_resume_cb(void) {
    *g_usb_available = true;
}
void
tud_umount_cb(void) {
    *g_usb_available = false;
}
void
tud_suspend_cb(bool remote_wakeup_en) {
    *g_usb_available = false;
}

uint32_t
init_usb_handler(usb_config_t *ctx) {
    g_usb_available = &ctx->available;
    return 0;
}
