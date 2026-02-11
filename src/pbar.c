#include "pbar.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>

static void _enable_virtual_terminal() {
    HANDLE h_out = GetStdHandle(STD_OUTPUT_HANDLE);
    if (h_out == INVALID_HANDLE_VALUE)
        return;

    DWORD dw_mode = 0;
    if (!GetConsoleMode(h_out, &dw_mode))
        return;

    dw_mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(h_out, dw_mode)
}
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif /* ifdef _WIN32 */

struct ProgressBar {
    int total;
    int digits;
    int width;
    time_t start_time;
};

static int _get_terminal_width() {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
        return csbi.srWindow.Right - csbi.srWindow.Left + 1;
    }
    return 80;
#else
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    return w.ws_col > 0 ? w.ws_col : 80;
#endif /* ifdef _WIN32 */
}

int num_digits(int num) {
    int digits = 0;
    while (num > 0) {
        digits++;
        num /= 10;
    }
    return digits;
}

ProgressBar *progress_init(int total) {
    ProgressBar *bar = malloc(sizeof(ProgressBar));
    if (!bar) {
        printf("Failure to create progress bar\n");
        exit(PBAR_INIT_FAILURE);
    }

#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    _enable_virtual_terminal();
#endif /* ifdef _WIN32 */

    bar->total = total;
    bar->digits = num_digits(total);
    bar->width = _get_terminal_width();
    bar->start_time = time(NULL);

    return bar;
}

static void _format_time(int seconds, char *out, size_t size) {
    int h = seconds / 3600;
    int m = (seconds % 3600) / 60;
    int s = seconds % 60;
    snprintf(out, size, "%02d:%02d:%02d", h, m, s);
}

void progress_update(const ProgressBar *bar, int current, const char *desc,
                     const char *postfix) {
    if (current > bar->total)
        current = bar->total;

    float ratio = (float)current / bar->total;
    int percent = (int)(ratio * 100);

    int bar_width = bar->width - 85;
    if (bar_width < 10)
        bar_width = 10;

    int filled = (int)(ratio * bar_width);

    /* Time calculations */
    time_t now = time(NULL);
    int elapsed = (int)difftime(now, bar->start_time);

    int eta = 0;
    double speed = 0.0;
    if (current > 0 && elapsed > 0) {
        eta = (int)((double)elapsed / current * (bar->total - current));
        speed = (double)current / elapsed;
    }

    char eta_str[16];
    _format_time(eta, eta_str, sizeof(eta_str));

    printf("\r\033[K");

    printf("%s%s%s ", COLOR_GREEN, desc, COLOR_RESET);

    printf("%s|", COLOR_BLUE);
    for (int i = 0; i < bar_width; i++) {
        if (i < filled)
            printf("█");
        else
            printf("░");
    }
    printf("|%s [%*d/%d] %3d%% ", COLOR_RESET, bar->digits, current, bar->total,
           percent);
    printf("|%s %sETA: %s (%.2f it/s) %s", COLOR_RESET, COLOR_YELLOW, eta_str,
           speed, COLOR_RESET);
    printf(" %s| %s%s", COLOR_GRAY, postfix, COLOR_RESET);

    fflush(stdout);
}

void progress_finish(ProgressBar *bar) {
    free(bar);
    printf("\n");
}
