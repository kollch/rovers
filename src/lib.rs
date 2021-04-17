//! A prototype rover environment for reinforcement learning.

#![warn(missing_docs)]

use std::rc::Rc;

use nalgebra as na;

/// A 2D point.
pub type Point = na::Point2<f64>;
/// A 2D vector.
pub type Vector = na::Vector2<f64>;
type State = na::MatrixXx1<f64>;

pub mod agents;
pub mod env;
pub mod sensors;

/// The type of a reward.
///
/// Determines how to [calculate](Reward::calculate()) the magnitude of the reward.
pub enum Reward {
    /// Sums the rewards of all [rovers](agents::Rover).
    Default,
    /// The difference between the default with and without a rover.
    Difference,
}

impl Reward {
    /// Determines how much to [reward](agents::Agent::reward()) a rover based on
    /// [agents](agents::Agent) around it and the [reward type](Reward).
    pub fn calculate(
        &self,
        id: usize,
        rovers: Vec<Rc<agents::Rover>>,
        pois: Vec<Rc<agents::Poi>>,
    ) -> f64 {
        use agents::{without_id, Agent, Poi, Rover};

        let default_calc = |rovers: Vec<Rc<Rover>>, pois: Vec<Rc<Poi>>| {
            pois.into_iter().map(|poi| poi.reward(&rovers)).sum()
        };
        match self {
            Reward::Default => default_calc(rovers, pois),
            Reward::Difference => {
                let reward = default_calc(rovers.clone(), pois.clone());
                let rovers = without_id(id, rovers);
                let reward_without_self = default_calc(rovers, pois);
                reward - reward_without_self
            }
        }
    }
}
