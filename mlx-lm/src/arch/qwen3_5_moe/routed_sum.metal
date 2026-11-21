uint c = thread_position_in_grid.x, t = thread_position_in_grid.y;
bfloat16_t sum = 0;
for (uint e = 0; e < 8; ++e) {
    uint j = t * 8 + e;
    bfloat16_t weighted = projected[order[j] * 2048 + c] * scores[j];
    sum = bfloat16_t(float(sum) + float(weighted));
}
result[t * 2048 + c] = sum;
