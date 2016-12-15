use rand::Rng;
use std::sync::Arc;
use std::sync::atomic::{Ordering, AtomicBool};

#[derive(Serialize, Deserialize)]
pub struct RectSomDefn {
  pub width: usize,
  pub height: usize,
}

#[derive(Serialize, Deserialize)]
pub struct RectSom {
  pub width: usize,
  pub height: usize,
  pub fields: Vec<Vec<f32>>,
}

impl RectSom {
  pub fn new_random<R: Rng>(width: usize, height: usize, bounding_box: &[(f32, f32)], mut rng: &mut R) -> RectSom {
    use rand::distributions::{Range, IndependentSample};
    let distributions = bounding_box.iter().map(|&(min, max)| Range::new(min, max)).collect::<Vec<_>>();

    RectSom {
      width: width,
      height: height,
      fields: (0..width*height).map(|_|
        distributions.iter().map(|&range| {
          range.ind_sample(&mut rng)
        }).collect()
      ).collect()
    }
  }

  pub fn field(&self, x: usize, y: usize) -> &[f32] {
    &self.fields[x + self.width * y][..]
  }

  pub fn delinearize_coord(&self, idx: usize) -> (usize, usize) {
    (idx % self.width, idx / self.height)
  }

  pub fn best_fitting_unit<Df: Copy + FnOnce(&[f32], &[f32]) -> f32>(&self, target: &[f32], dist: Df) -> (usize, usize) {
    let mut min_idx = 0;
    let mut min_dist = dist(&self.fields[min_idx][..], target);
    for it in 1..self.fields.len() {
      let d = dist(&self.fields[it][..], target);
      if d < min_dist {
        min_idx = it;
        min_dist = d;
      }
    }
    self.delinearize_coord(min_idx)
  }

  pub fn nudge_weights<Nf: Copy + FnOnce(isize, isize, f32) -> f32>(&mut self, at: (usize, usize), radius: f32, target: &[f32], neighbourhood_fn: Nf, rate: f32) -> f32 {
    let (at_xi, at_yi) = (at.0 as isize, at.1 as isize);
    let mut total_magnitude = 0f32;
    for it in 0..self.fields.len() {
      let (x, y) = self.delinearize_coord(it);
      let (xi, yi) = (x as isize, y as isize);
      let nudge_scale = neighbourhood_fn(xi - at_xi, yi - at_yi, radius);
      let mut this_magnitude = 0f32;
      for (jt, w) in self.fields[it].iter_mut().enumerate() {
        let new_w = target[jt] * nudge_scale * rate + *w * (1.0 - nudge_scale * rate);
        let diff = *w - new_w;
        this_magnitude += diff * diff;
        *w = new_w;
      }
      total_magnitude += this_magnitude.sqrt() / target.len() as f32;
    }
    total_magnitude / self.fields.len() as f32
  }

  #[allow(dead_code)]
  pub fn apply_diff(&mut self, diff: &[Vec<f32>]) {
    for it in 0..self.fields.len() {
      for (jt, w) in self.fields[it].iter_mut().enumerate() {
        *w += diff[it][jt];
      }
    }
  }
}

#[derive(Serialize, Deserialize)]
pub struct TrainConfig {
  pub train_rate: f32,
  pub stability_threshold: f32,
  pub stability_duration: usize,
  pub initial_radius: f32,
  pub radius_decay: f32,
  pub min_radius: Option<f32>,
  pub max_epochs: usize,
  pub neighbourhood: String,
}

pub fn gaussian(dx: isize, dy: isize, radius: f32) -> f32 {
  let (dx, dy) = (dx as f32, dy as f32);
  (-(dx*dx + dy*dy)/(2.0*radius*radius)).exp()
}

pub fn sinc_sq(dx: isize, dy: isize, radius: f32) -> f32 {
  if dx == 0 && dy == 0 {
    return 1.0;
  }
  let (dx, dy) = (dx as f32, dy as f32);
  let dist = (dx*dx + dy*dy).sqrt() / (radius * 2.0);
  let sinc = dist.sin() / dist;
  sinc
}

pub fn expon(dx: isize, dy: isize, radius: f32) -> f32 {
  let (dx, dy) = (dx as f32, dy as f32);
  let dist = (dx*dx + dy*dy).sqrt() / radius;
  (-dist).exp()
}

pub fn train<R: Rng, Df: Copy + FnOnce(&[f32], &[f32]) -> f32>(
    examples: &[Vec<f32>],
    som: &mut RectSom,
    dist_fn: Df,
    rng: &mut R,
    conf: &TrainConfig,
    stop_flag: Arc<AtomicBool>) {
  let mut stability = 0usize;
  let mut radius = conf.initial_radius;
  let mut epoch = 0usize;

  while stop_flag.load(Ordering::SeqCst) &&
      epoch < conf.max_epochs &&
      stability < conf.stability_duration {
    epoch += 1;
    if epoch % 10 == 0 {
      println!("#{}", epoch);
    }
    let ex = &rng.choose(examples).unwrap()[..];
    let bfu = som.best_fitting_unit(ex, dist_fn);
    let diff = som.nudge_weights(bfu, radius, ex, match &conf.neighbourhood[..] {
      "gaussian" => gaussian,
      "sinc_sq" => sinc_sq,
      "expon" => expon,
      _ => unreachable!(),
    }, conf.train_rate);
    if diff < conf.stability_threshold {
      stability += 1;
    } else {
      stability = 0;
    }
    radius *= conf.radius_decay;
    if let Some(min) = conf.min_radius {
      if radius < min {
        radius = min;
      }
    }
  }
}