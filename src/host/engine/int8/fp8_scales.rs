//! Static FP8 activation scales for all layers and projections.
const BLOB: &[u8] = include_bytes!("fp8_scales.bin");
pub(super) fn load() -> Result<[[f32; 4]; 48], Box<dyn std::error::Error>> {
    let v: serde_json::Value = serde_json::from_str(include_str!("fp8_scales.json"))?;
    if v["meta"]["calibration"]["sha256"] != "7f5b6419eb58f71cc899f88015c938d71fd2e1e74aaa0ff42e7c3b320bafef83" || BLOB.len() != 768 {
        return Err("FP8 scales have incorrect calibration identity or geometry".into());
    }
    let sites=v["scales"].as_array().ok_or("FP8 scale metadata missing")?;
    if sites.len()!=192 { return Err("FP8 scale metadata must have 192 sites".into()); }
    let mut result=[[0f32;4];48];
    for (i,(row,bytes)) in sites.iter().zip(BLOB.chunks_exact(4)).enumerate() {
        let scale=f32::from_le_bytes(bytes.try_into()?);
        if !scale.is_finite() || scale<=0. || row["layer"]!=i/4 || row["site"]!=["qkv","o","mlp","down"][i%4]
            || row["scale_f32_hex"]!=format!("0x{:08x}",scale.to_bits())
            || row["scale"].as_f64()!=Some(scale as f64)
            || row["calib_amax_max"].as_f64()!=Some((scale*256.) as f64) {
            return Err(format!("FP8 static scale entry {i} differs from its neutral calibration binding").into());
        }
        result[i/4][i%4]=scale;
    }
    Ok(result)
}
