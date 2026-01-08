#ifndef PBAR_H
#define PBAR_H

#define COLOR_RESET "\033[0m"
#define COLOR_GREEN "\033[38;5;82m"
#define COLOR_BLUE "\033[38;5;75m"
#define COLOR_GRAY "\033[38;5;245m"
#define COLOR_YELLOW "\033[38;5;214m"

#define PBAR_INIT_FAILURE 1

typedef struct ProgressBar ProgressBar;

ProgressBar *progress_init(int total);
void progress_update(const ProgressBar *bar, int current, const char *desc,
                     const char *postfix);
void progress_finish();

#endif // !PBAR_H
