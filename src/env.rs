//! Environment utilities to provide bounds on the world the agents interact in.

use super::{
    agents::{Agent, Poi, Rover},
    Point, Rc, State, Vector,
};
use nalgebra as na;

/// An environment for [rovers](Rover) and [POIs](Poi).
///
/// Contains a boundary to the world.
pub struct Environment<T: EnvInit> {
    init_policy: T,
    rovers: Vec<Rc<Rover>>,
    pois: Vec<Rc<Poi>>,
    /// The number of rovers.
    pub num_rovers: usize,
    /// The number of POIs.
    pub num_pois: usize,
    /// The size of the environment, in (width, height).
    pub size: (f64, f64),
}

impl<T: EnvInit> Environment<T> {
    /// Constructs a new `Environment<T>`.
    ///
    /// Expects rover and POI [`Vec`]s with [`Rc`]s that haven't been [cloned](Rc::clone()) yet;
    /// a singular reference is necessary to mutate the rovers and POIs inside of them.
    pub fn new(
        init_policy: T,
        rovers: Vec<Rc<Rover>>,
        pois: Vec<Rc<Poi>>,
        size: (f64, f64),
    ) -> Self {
        let num_rovers = rovers.len();
        let num_pois = pois.len();
        Environment {
            init_policy,
            rovers,
            pois,
            num_rovers,
            num_pois,
            size,
        }
    }

    /// Has rovers perform a singular action apiece.
    ///
    /// This function can be thought of as a "frame" or timestep through the environment.
    pub fn step(&mut self, actions: Vec<Vector>) -> (Vec<State>, Vec<f64>) {
        for (i, mut rover) in self.rovers.iter_mut().enumerate() {
            // Take actions
            match Rc::get_mut(&mut rover) {
                Some(r) => r.act(actions[i]),
                None => eprintln!("Warning: More than one reference to rover - not acting."),
            }
        }
        // Bound positions
        self.clamp_positions();
        // Return next observations and rewards
        self.status()
    }

    /// Resets rover and POI fields.
    ///
    /// Specifically, calls the [`Agent`]-specific function [`Agent::reset()`] for all
    /// rovers and POIs in the environment.
    pub fn reset(&mut self) -> (Vec<State>, Vec<f64>) {
        // Clear agents
        for mut rover in self.rovers.iter_mut() {
            Rc::get_mut(&mut rover)
                .expect("Error: More than one reference to rover.")
                .reset();
        }
        // Reset POIs
        for mut poi in self.pois.iter_mut() {
            Rc::get_mut(&mut poi)
                .expect("Error: More than one reference to POI.")
                .reset();
        }
        // Initialize
        self.init_policy.init(
            self.rovers.iter_mut().collect(),
            self.pois.iter_mut().collect(),
            self.size,
        );
        self.status()
    }

    /// Clamps positions for all rovers within the environment's bounds.
    ///
    /// Iterates through all rovers and bounds their positions to the bounds of the environment.
    fn clamp_positions(&mut self) {
        for mut rover in self.rovers.iter_mut() {
            let clamped = na::clamp(
                rover.pos(),
                Point::origin(),
                Point::new(self.size.0, self.size.1),
            );
            Rc::get_mut(&mut rover)
                .expect("Error: More than one reference to rover.")
                .set_pos(clamped.x, clamped.y);
        }
    }

    /// Returns the states and rewards of each rover.
    ///
    /// The *i*th element of each returning vector corresponds to the *i*th rover.
    fn status(&self) -> (Vec<State>, Vec<f64>) {
        let mut states = Vec::with_capacity(self.num_rovers);
        let mut rewards = Vec::with_capacity(self.num_rovers);
        // Observations and rewards
        for rover in self.rovers.iter() {
            states.push(rover.scan(self.rovers.clone(), self.pois.clone()));
            rewards.push(rover.get_reward(self.rovers.clone(), self.pois.clone()));
        }
        (states, rewards)
    }
}

/// An [environment](Environment) initialization type, randomizing locations.
///
/// Sets random locations for [agents](Agent) in the environment with the [`EnvInit`] trait.
pub struct EnvRand;

impl EnvInit for EnvRand {
    /// Sets all [rover](Rover) positions to the origin.
    fn init_rovers(&self, rovers: Vec<&mut Rc<Rover>>, _size: (f64, f64)) {
        for mut rover in rovers {
            Rc::get_mut(&mut rover)
                .expect("Error: More than one reference to rover.")
                .set_pos(0.0, 0.0);
        }
    }

    /// Sets all [POI](Poi) positions to the origin.
    fn init_pois(&self, pois: Vec<&mut Rc<Poi>>, _size: (f64, f64)) {
        for mut poi in pois {
            Rc::get_mut(&mut poi)
                .expect("Error: More than one reference to POI.")
                .set_pos(0.0, 0.0);
        }
    }
}

/// An [environment](Environment) initialization type, setting [POIs](Poi) to the corners of the
/// environment.
///
/// [Rovers](Rover) are set near the center of the environment and POIs are set near the corners.
/// This struct uses the [`EnvInit`] trait to set locations.
pub struct EnvCorners;

impl EnvInit for EnvCorners {
    /// Sets all rover positions oriented around the center of the environment.
    fn init_rovers(&self, rovers: Vec<&mut Rc<Rover>>, size: (f64, f64)) {
        let span = f64::min(size.0, size.1);
        let start = 1.0;
        let end = span - 1.0;
        let rad = span / f64::sqrt(3.0) / 2.0;
        let center = (start + end) / 2.0;
        for (i, mut rover) in rovers.into_iter().enumerate() {
            let offset1 = (i as f64 / 4.0) % (center - rad);
            let offset2 = (i as f64 / (4.0 * center - rad)) % (center - rad);
            let (x, y) = match i % 4 {
                0 => (center - 1.0 - offset1, center - offset2),
                1 => (center - 1.0 + offset2, center - 1.0 + offset1),
                2 => (center + 1.0 + offset1, center + offset2),
                3 => (center - offset2, center + 1.0 - offset1),
                _ => unreachable!(),
            };
            Rc::get_mut(&mut rover)
                .expect("Error: More than one reference to rover.")
                .set_pos(x, y);
        }
    }

    /// Sets the POIs in the corners of the environment.
    fn init_pois(&self, pois: Vec<&mut Rc<Poi>>, size: (f64, f64)) {
        let span = f64::min(size.0, size.1);
        let start = 0.0;
        let end = span - 1.0;
        for (i, mut poi) in pois.into_iter().enumerate() {
            let offset = f64::trunc(i as f64 / 4.0);
            let (x, y) = match i % 4 {
                0 => (start + offset, start + offset),
                1 => (end - offset, start + offset),
                2 => (start + offset, end - offset),
                3 => (end - offset, end - offset),
                _ => unreachable!(),
            };
            Rc::get_mut(&mut poi)
                .expect("Error: More than one reference to POI.")
                .set_pos(x, y);
        }
    }
}

/// Provides a set of functions that initialize [rovers](Rover) and [POIs](Poi) to new
/// locations.
pub trait EnvInit {
    /// Initializes provided [rovers](Rover) with new locations.
    fn init_rovers(&self, rovers: Vec<&mut Rc<Rover>>, size: (f64, f64));
    /// Initializes provided [POIs](Poi) with new locations.
    fn init_pois(&self, pois: Vec<&mut Rc<Poi>>, size: (f64, f64));
    /// Initializes provided [rovers](Rover) and [POIs](Poi) with new locations.
    fn init(&self, rovers: Vec<&mut Rc<Rover>>, pois: Vec<&mut Rc<Poi>>, size: (f64, f64)) {
        self.init_rovers(rovers, size);
        self.init_pois(pois, size);
    }
}
