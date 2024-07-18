

typedef struct {
    double a;
    int b;
} my_struct;

void mprintf(const char*, int);

int diff(my_struct before, my_struct after) {
    mprintf("before.a: %d\n", before.a);
    int x = (after.a - before.a)*1000;
    int b =(after.b - before.b);
    return x + b;
}

// int main() {
//     my_struct before = {1, 2};
//     my_struct after = {3, 4};
//     return diff(before, after);
// }

