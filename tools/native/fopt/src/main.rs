//! Parse, split and rebuild Opt 0.9.0 device-function images with the vendor ABI.
//! usage: foptool info <img> | split <img> <dir> | join <dir> <out>
use furiosa_opt_abi::image::{Depth, Image, Memory, Parts, Profile, Slot, Span, Stack, Weight};
use serde_json::{json, Value};
use std::{env, fs, path::Path};

fn mem(m: Memory) -> &'static str { match m { Memory::Dram => "dram", Memory::Sram => "sram" } }
fn unmem(s: &str) -> Memory { if s == "dram" { Memory::Dram } else { Memory::Sram } }

fn meta(img: &Image) -> Value {
    let d = img.profile().depth;
    json!({
        "chips": img.chips(), "pes": img.pes(),
        "tasks": img.tasks().iter().map(|c| c.iter().map(|t| t.len()).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "weight": img.weight().map(|w| json!({"kind": mem(w.kind), "len": w.bytes.len()})),
        "stacks": img.stacks().iter().map(|s| json!({"kind": mem(s.kind), "size": s.size})).collect::<Vec<_>>(),
        "slots": img.slots().iter().map(|s| json!({"kind": mem(s.kind), "size": s.size})).collect::<Vec<_>>(),
        "inputs": img.inputs(), "outputs": img.outputs(),
        "staging": img.staging().expect("staging").len(),
        "sram_end": img.sram_end(),
        "profile": {"depth": {"info": d.info, "debug": d.debug, "trace": d.trace}, "spans": img.profile().spans.iter().map(|c| c.iter().map(|s| json!([s.name, s.begin, s.end])).collect::<Vec<_>>()).collect::<Vec<_>>()},
    })
}

fn main() {
    let a: Vec<String> = env::args().collect();
    match a[1].as_str() {
        "info" => {
            let b = fs::read(&a[2]).unwrap();
            let img = Image::parse(&b).expect("parse");
            let mut m = meta(&img);
            let n: usize = m["profile"]["spans"].as_array().unwrap().iter().map(|c| c.as_array().unwrap().len()).sum();
            m["profile"]["spans"] = json!(format!("{} spans", n));
            println!("{}", serde_json::to_string_pretty(&m).unwrap());
        }
        "split" => {
            let b = fs::read(&a[2]).unwrap();
            let img = Image::parse(&b).expect("parse");
            let d = Path::new(&a[3]); fs::create_dir_all(d).unwrap();
            for (ci, c) in img.tasks().iter().enumerate() { for (k, t) in c.iter().enumerate() { fs::write(d.join(format!("task_{ci}_{k}.bin")), t).unwrap(); } }
            if let Some(w) = img.weight() { fs::write(d.join("weight.bin"), w.bytes).unwrap(); }
            fs::write(d.join("parts.json"), serde_json::to_string_pretty(&meta(&img)).unwrap()).unwrap();
        }
        "join" => {
            let d = Path::new(&a[2]);
            let m: Value = serde_json::from_str(&fs::read_to_string(d.join("parts.json")).unwrap()).unwrap();
            let chunks = m["tasks"].as_array().unwrap();
            let owned: Vec<Vec<Vec<u8>>> = chunks.iter().enumerate().map(|(ci, c)| (0..c.as_array().unwrap().len()).map(|k| fs::read(d.join(format!("task_{ci}_{k}.bin"))).unwrap()).collect()).collect();
            let wbytes = if m["weight"].is_null() { None } else { Some(fs::read(d.join("weight.bin")).unwrap()) };
            let names: Vec<Vec<(String, u16, u16)>> = m["profile"]["spans"].as_array().unwrap().iter().map(|c| c.as_array().unwrap().iter().map(|s| (s[0].as_str().unwrap().to_string(), s[1].as_u64().unwrap() as u16, s[2].as_u64().unwrap() as u16)).collect()).collect();
            let dp = &m["profile"]["depth"];
            let parts = Parts {
                chips: m["chips"].as_u64().unwrap() as u8, pes: m["pes"].as_u64().unwrap() as u8,
                tasks: owned.iter().map(|c| c.iter().map(|t| t.as_slice()).collect()).collect(),
                weight: wbytes.as_ref().map(|w| Weight { kind: unmem(m["weight"]["kind"].as_str().unwrap()), bytes: w.as_slice() }),
                stacks: m["stacks"].as_array().unwrap().iter().map(|s| Stack { kind: unmem(s["kind"].as_str().unwrap()), size: s["size"].as_u64().unwrap() }).collect(),
                slots: m["slots"].as_array().unwrap().iter().map(|s| Slot { kind: unmem(s["kind"].as_str().unwrap()), size: s["size"].as_u64().unwrap() }).collect(),
                inputs: m["inputs"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect(),
                outputs: m["outputs"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect(),
                profile: Profile {
                    depth: Depth { info: dp["info"].as_u64().unwrap() as u32, debug: dp["debug"].as_u64().unwrap() as u32, trace: dp["trace"].as_u64().unwrap() as u32 },
                    spans: names.iter().map(|c| c.iter().map(|(n, b, e)| Span { name: n.as_str(), begin: *b, end: *e }).collect()).collect(),
                },
                sram_end: m["sram_end"].as_u64().unwrap(),
            };
            let img = Image::new(parts).expect("Image::new rejected the parts");
            fs::write(&a[3], img.encode().expect("encode")).unwrap();
        }
        _ => panic!("usage: foptool info <img> | split <img> <dir> | join <dir> <out>"),
    }
}
