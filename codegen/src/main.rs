#![feature(portable_simd)]
use regex::Captures;
use regex::Regex;
use std::fs;

fn read_all() {
    let pattern = r"Multiply([\S\s]*?)T";
    let re = Regex::new(pattern).unwrap();
    let file_path = "data/algorithms.txt";
    let contents = fs::read_to_string(file_path).expect("Should have been able to read the file");
    let caps = re.captures_iter(&contents);
    println!("{}", "#![allow(unused_parens)]");
    println!("{}", "use std::simd::i32x16;");
    println!("");
    let mut count = 0;
    for c in caps {
        let kl = c.get(0).map_or("", |m| m.as_str()).to_string();
        convert(&kl);
        if count > 24 {
            break;
        }
        count += 1;
    }
}

fn main() {
    read_all();
}

fn convert(string: &String) {
    let sd = string.lines().next().unwrap();
    let hs = r"h1((.+\n)+)(c_1_1|c11) ";
    let re = Regex::new(hs).unwrap();
    let caps = re.captures(string).unwrap();
    let text1 = caps
        .get(0)
        .map_or("", |m| m.as_str())
        .trim_end_matches("c11 ")
        .trim_end_matches("c_1_1 ");

    let re = Regex::new(r".*= ").unwrap();
    let mut lefts = vec![];
    let mut rights = vec![];
    for line in text1.lines() {
        let parts: Vec<&str> = line.split(r" * ").collect();
        let left = re.replace_all(parts[0], "").to_string();
        let right = parts[1].to_string();
        lefts.push(left);
        rights.push(right);
    }
    let end_mult = caps.get(0).unwrap().end();
    let func_name = sd
        .to_lowercase()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("+", "_");

    let mut inputs = vec![];
    let mut counter = 0;
    for lll in string.lines() {
        if lll.contains("ravel") {
            let input_split: Vec<_> = lll.split(" = ").collect();
            let vars = input_split[0].to_string();
            let _all_vars: Vec<_> = vars.split(", ").map(|s| s.to_string()).collect();
            // all_vars.extend(_all_vars.clone());
            let key = if counter == 0 { "a" } else { "b" };
            inputs.push((_all_vars, key));
            counter += 1;
        }
    }
    let ars = format!(
        "a: [i32; {}], b: [i32; {}]",
        inputs[0].0.len(),
        inputs[1].0.len()
    );

    let mut inputers = vec![];
    for _in in inputs {
        let (_all_vars, key) = _in;
        inputers.push(format!("\tlet [{}] = {};", _all_vars.join(", "), key));
    }
    let simd_size = 16;
    let binding = lefts.clone();
    let left_chunks = binding.chunks(simd_size);

    let mut lchunk = vec![];
    for w in left_chunks {
        let mut x = w.clone().to_vec();
        x.resize(simd_size, "0".to_string());
        lchunk.push(format!("\t\ti32x{}::from([{}]),", simd_size, x.join(", ")));
    }

    let binding = rights.clone();
    let right_chunks = binding.chunks(simd_size);
    let mut rchunk = vec![];
    for w in right_chunks {
        let mut x = w.clone().to_vec();
        x.resize(simd_size, "0".to_string());
        rchunk.push(format!("\t\ti32x{}::from([{}]),", simd_size, x.join(", ")));
    }

    let mut ckeys = vec![];
    let p = string.split_at(end_mult);
    let end = format!("c11{};", p.1);
    let mut cs = vec![];
    for linex in end.lines() {
        // skip last line?
        if linex.starts_with("C =") {
            continue;
        }
        let c_key = linex.split("=").collect::<Vec<_>>()[0].trim();
        // replace hs with relative index
        let hfinder = Regex::new(r"h[0-9]+").unwrap();
        let s2 = hfinder.replace_all(linex, |cap: &Captures| {
            let item = cap.get(0).unwrap().as_str().to_string();
            // dbg!(item.clone());
            let number = item.trim_start_matches("h").parse::<i32>().unwrap() - 1;
            let which_simd_arr = number / simd_size as i32;
            let simd_specific_index = number - (which_simd_arr * simd_size as i32);
            let result = format!("hs[{which_simd_arr}][{simd_specific_index}]");
            result
        });
        cs.push(c_key);
        ckeys.push(format!("\tlet {};", s2));
    }
    let res_count = cs.len();

    println!("pub fn {}({}) -> [i32; {}] {{", func_name, ars, res_count);
    for i in inputers {
        println!("{}", i);
    }
    println!("\t{}", "let lefts = [");
    for i in lchunk.clone() {
        println!("{}", i);
    }
    println!("\t{}", "];");
    println!("\t{}", "let rights = [");
    for i in rchunk {
        println!("{}", i);
    }
    println!("\t{}", "];");
    println!("\t{}", "let hs = [");
    for i in 0..lchunk.len() {
        println!("\t\tlefts[{}] * rights[{}],", i, i);
    }
    println!("\t{}", "];");
    for i in ckeys {
        println!("{}", i);
    }
    println!("\n\treturn [{}];\n}}", cs.join(", "));
}
