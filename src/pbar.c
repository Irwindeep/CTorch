#include "pbar.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    SetConsoleMode(h_out, dw_mode);
}
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif /* ifdef _WIN32 */

struct ProgressBar {
    int total;
    int digits;
    int width;
    clock_t start_time;
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
    bar->start_time = clock();

    return bar;
}

static void _format_time(int seconds, char *out, size_t size) {
    int h = seconds / 3600;
    int m = (seconds % 3600) / 60;
    int s = seconds % 60;
    snprintf(out, size, "%02d:%02d:%02d", h, m, s);
}

static int visible_len(const char *s) {
    int len = 0;
    while (*s) {
        if (*s == '\033') {
            while (*s && *s != 'm')
                s++;
            if (*s)
                s++;
        } else {
            len++;
            s++;
        }
    }
    return len;
}

void progress_update(ProgressBar *bar, int current, const char *desc,
                     const char *postfix) {
    if (current > bar->total)
        current = bar->total;

    int term_width = _get_terminal_width();
    float ratio = (float)current / bar->total;
    int percent = (int)(ratio * 100.0f);

    clock_t now = clock();
    double elapsed = (double)(now - bar->start_time) / CLOCKS_PER_SEC;

    double speed = 0.0;
    int eta = 0;

    if (current > 0 && elapsed > 0) {
        speed = (double)current / elapsed;
        eta = (int)((bar->total - current) / speed);
    }

    char eta_str[16];
    _format_time(eta, eta_str, sizeof(eta_str));

    char left[256];
    snprintf(left, sizeof(left), "%s%s%s | %s%s%s", COLOR_GREEN, desc,
             COLOR_RESET, COLOR_GRAY, postfix, COLOR_RESET);

    char right[256];
    snprintf(right, sizeof(right), "[%*d/%d] %.2f it/s %s%s%s ", bar->digits,
             current, bar->total, speed, COLOR_YELLOW, eta_str, COLOR_RESET);

    int left_len = visible_len(left);
    int right_len = visible_len(right);

    int spacing = 4;
    int percent_len = 4;
    int fixed_chars = 4;

    int bar_width =
        term_width - left_len - right_len - percent_len - fixed_chars - 1;
    if (bar_width < 10)
        bar_width = 10;
    if (bar_width > 60)
        bar_width = 60;

    printf("\r\033[K");

    printf("%s", left);

    int used_without_padding =
        left_len + right_len + percent_len + fixed_chars + bar_width;

    int padding = term_width - used_without_padding;

    if (padding < 1)
        padding = 1;

    printf("%*s", padding, "");

    printf("%s", right);
    printf("%s[", COLOR_BLUE);

    int pos = (int)(ratio * (bar_width - 1));

    if (current == bar->total) {
        for (int i = 0; i < bar_width; i++)
            printf("-");
    } else {
        for (int i = 0; i < bar_width; i++) {
            if (i < pos)
                printf("-");
            else if (i == pos) {
                printf(i % 2 == 0 ? "C" : "c");
            } else if (i % 2 == 0)
                printf("o");
            else
                printf(" ");
        }
    }

    printf("]%s", COLOR_RESET);
    printf(" %3d%%", percent);

    fflush(stdout);
}

void progress_finish(ProgressBar *bar) {
    free(bar);
    printf("\n");
}
