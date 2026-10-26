The intended runtime device is `cuda:0`. Both files move a tensor
to the accelerator; audit whether the wrapper change alters the
computation in the intended runtime.
