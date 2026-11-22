const int B  = rel_shape[0];
const int LQ = rel_shape[1];
const int H  = rel_shape[2];
const int S  = kshape_shape[2];
uint j  = thread_position_in_grid.x;
uint i  = thread_position_in_grid.y;
uint bh = thread_position_in_grid.z;
if ((int)i >= LQ || (int)j >= S || (int)bh >= B * H) return;
uint b = bh / H, h = bh % H;
int dist = ((int)i + (S - LQ)) - (int)j;
T val;
if (dist < 0) {
    val = (T)(-1e30f);
} else if (SLIDING > 0 && dist >= (int)SLIDING) {
    val = (T)(-1e30f);
} else if (dist < (int)REL_EXTENT) {
    float acc = 0.0f;
    const size_t rbase = (size_t)b * rel_strides[0]
        + (size_t)i * rel_strides[1] + (size_t)h * rel_strides[2];
    for (uint d = 0; d < D_REL; ++d)
        acc += (float)rel[rbase + (size_t)d * rel_strides[3]]
             * (float)proj[(size_t)d * proj_strides[0]
                           + (size_t)dist * proj_strides[1]];
    val = (T)acc;
} else {
    val = (T)0;
}
out[(((size_t)b * H + h) * LQ + i) * S + j] = val;
