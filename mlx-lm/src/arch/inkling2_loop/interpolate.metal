uint gx = thread_position_in_grid.x;
uint y_out = thread_position_in_grid.y;
uint b = thread_position_in_grid.z;

int in_h = x_shape[1];
int in_w = x_shape[2];
int channels = x_shape[3];
int out_h = iy_shape[0];
int taps_y = iy_shape[1];
int out_w = ix_shape[0];
int taps_x = ix_shape[1];

if (gx >= (uint)(out_w * channels) || y_out >= (uint)out_h)
    return;

int x_out = gx / channels;
size_t y_tap = (size_t)y_out * taps_y;
size_t x_tap = (size_t)x_out * taps_x;
size_t input_base = (size_t)b * in_h * in_w * channels + (gx % channels);

float result = 0.0f;
for (int a = 0; a < taps_y; a++) {
    size_t row = input_base + (size_t)iy[y_tap + a] * in_w * channels;
    float row_result = 0.0f;
    for (int c = 0; c < taps_x; c++) {
        row_result += x[row + (size_t)ix[x_tap + c] * channels] * wx[x_tap + c];
    }
    result += row_result * wy[y_tap + a];
}

out[((size_t)b * out_h + y_out) * out_w * channels + gx] = result;
