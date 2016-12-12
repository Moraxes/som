#![feature(proc_macro)]

// extern crate bincode;
// extern crate byteorder;
extern crate clap;
extern crate ctrlc;
extern crate image;
// extern crate nalgebra;
extern crate rand;
// extern crate rayon;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json as sj;
extern crate palette;

mod som;
mod program_args;

use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{Ordering, AtomicBool};

fn main() {
  let args = program_args::get();

  match args.subcommand() {
    ("train-img", m) => train_img(m.unwrap()),
    _ => unreachable!(),
  }
}

fn gamma_expand(x: f32) -> f32 {
  if x <= 0.04045 {
    x / 12.92
  } else {
    ((x + 0.055) / 1.055).powf(2.4)
  }
}

fn gamma_compress(x: f32) -> f32 {
  if x <= 0.0031308 {
    x * 12.92
  } else {
    1.055 * x.powf(1.0 / 2.4) - 0.055
  }
}

fn sq_euclidean_dist(x: &[f32], y: &[f32]) -> f32 {
  x.iter().zip(y).map(|(x, y)| x - y).map(|z| z * z).sum()
}

fn train_img<'a>(args: &clap::ArgMatches<'a>) {
  use image::Pixel;
  use rand::SeedableRng;

  let img = match image::open(args.value_of("input").unwrap()) {
    Ok(img) => img,
    Err(err) => panic!("error while loading image: {}", err.description())
  }.to_rgb();

  let img_f32 = img.pixels().map(|p| {
    use palette::*;

    let rgb = p.to_rgb();
    let r = gamma_expand(rgb[0] as f32 / 255.0);
    let g = gamma_expand(rgb[1] as f32 / 255.0);
    let b = gamma_expand(rgb[2] as f32 / 255.0);
    let lab = Lab::from_rgb(Rgb::new(r, g, b));
    vec![
      lab.l,
      lab.a,
      lab.b,
    ] 
  }).collect::<Vec<Vec<f32>>>();

  let conf: som::TrainConfig = {
    use std::fs::File;
    match File::open(args.value_of("config").unwrap()) {
      Ok(file) => sj::from_reader(file).unwrap(),
      Err(_) => panic!("no config file"),
    }
  };

  let net_defn: som::RectSomDefn = {
    use std::fs::File;
    match File::open(args.value_of("net_defn").unwrap()) {
      Ok(file) => sj::from_reader(file).unwrap(),
      Err(_) => panic!("no network definition file"),
    }
  };

  let mut rng = rand::XorShiftRng::from_seed(rand::random());
  let mut net = som::RectSom::new_random(net_defn.width, net_defn.height, &vec![(0.0, 100.0), (-128.0, 127.0), (-128.0, 127.0)], &mut rng);

  let learning = Arc::new(AtomicBool::new(true));
  let l = learning.clone();
  ctrlc::set_handler(move || {
    l.store(false, Ordering::SeqCst);
  });

  som::train(&img_f32, &mut net, sq_euclidean_dist, &mut rng, &conf, learning);

  let map_img = image::RgbImage::from_fn(net.width as u32, net.height as u32,
      |x, y| {
        use palette::*;
        // let mut field = net.field(x as usize, y as usize).iter().map(|&c| gamma_compress(c) * 255.0).map(|c| c as u8);
        // let r = field.next().unwrap();
        // let g = field.next().unwrap();
        // let b = field.next().unwrap();
        let mut field = net.field(x as usize, y as usize).iter();
        let rgb = Rgb::from_lab(Lab::new(
          *field.next().unwrap(),
          *field.next().unwrap(),
          *field.next().unwrap()
        ));
        image::Rgb {
          data: [
            (gamma_compress(rgb.red) * 255.0) as u8,
            (gamma_compress(rgb.green) * 255.0) as u8,
            (gamma_compress(rgb.blue) * 255.0) as u8,
          ],
        }
      });
  match map_img.save(args.value_of("image").unwrap()) {
    Err(err) => panic!("failed to save image: {}", err.description()),
    _ => {},
  };
}
