// Adapted from Apple mlx-lm's gated_delta_step, MIT license.
uint lane = thread_position_in_grid.x, row = thread_position_in_grid.y, h = thread_position_in_grid.z;
uint base = (h * 128 + row) * 128 + lane * 4;
float mem[4];
for (uint i=0; i<4; ++i) mem[i] = state[base+i];
for (int t=0; t<q_shape[0]; ++t) {
    uint col = (t*16+h/2)*128 + lane*4, head = t*32+h;
    float pred = 0;
    for (uint i=0; i<4; ++i) { mem[i] *= g[head]; pred += mem[i]*k[col+i]; }
    for (uint d=1; d<=16; d*=2) pred += simd_shuffle_xor(pred,d);
    float delta = (float(v[head*128+row])-pred)*beta[head], y=0;
    for (uint i=0; i<4; ++i) { mem[i] += k[col+i]*delta; y += mem[i]*q[col+i]; }
    for (uint d=1; d<=16; d*=2) y += simd_shuffle_xor(y,d);
    if (!lane) out[head*128+row] = bfloat16_t(y);
}
for (uint i=0; i<4; ++i) next[base+i] = mem[i];
