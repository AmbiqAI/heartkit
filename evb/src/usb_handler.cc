// Handle TinyUSB events

#include "usb_handler.h"
#include "ns_ambiqsuite_harness.h"
#include "ns_usb.h"

bool volatile *g_usb_available = nullptr;

void
tud_mount_cb(void) {
    if (g_usb_available != nullptr) {
        *g_usb_available = true;
    }
}
void
tud_resume_cb(void) {
    if (g_usb_available != nullptr) {
        *g_usb_available = true;
    }
}
void
tud_umount_cb(void) {
    if (g_usb_available != nullptr) {
        *g_usb_available = false;
    }
}
void
tud_suspend_cb(bool remote_wakeup_en) {
    if (g_usb_available != nullptr) {
        *g_usb_available = false;
    }
}

void
usb_update_state() {
    if (g_usb_available != nullptr) {
        *g_usb_available = tud_mounted();
    }
}

uint32_t
init_usb_handler(usb_config_t *ctx) {
    g_usb_available = &ctx->available;
    usb_update_state();
    return 0;
}
