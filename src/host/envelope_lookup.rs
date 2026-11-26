//! Reconstruction of frozen byte tables from an LZ4 block.
//! The returned array is the original include_bytes! value. There is no runtime
//! decompressor, crate dependency, build script, or startup work.
pub(crate) const fn unpack<const N: usize>(input: &[u8]) -> [u8; N] {
    let mut out = [0u8; N];
    unpack_into(input, &mut out);
    out
}

pub(crate) const fn unpack_into(input: &[u8], out: &mut [u8]) {
    let mut ip = 0;
    let mut op = 0;
    while ip < input.len() {
        let token = input[ip];
        ip += 1;
        let mut literals = (token >> 4) as usize;
        if literals == 15 {
            loop {
                let n = input[ip] as usize;
                ip += 1;
                literals += n;
                if n != 255 { break; }
            }
        }
        let end = op + literals;
        while op < end {
            out[op] = input[ip];
            ip += 1;
            op += 1;
        }
        if ip == input.len() { break; }
        let offset = input[ip] as usize | ((input[ip + 1] as usize) << 8);
        ip += 2;
        assert!(offset > 0 && offset <= op);
        let mut count = (token & 15) as usize + 4;
        if token & 15 == 15 {
            loop {
                let n = input[ip] as usize;
                ip += 1;
                count += n;
                if n != 255 { break; }
            }
        }
        let end = op + count;
        while op < end {
            out[op] = out[op - offset];
            op += 1;
        }
    }
    assert!(op == out.len() && ip == input.len());
}
