//! Sensors used by [rovers](super::agents::Rover).

use super::{
    agents::{Agent, Poi, Rover},
    Point, State,
};
use nalgebra as na;
use std::rc::Rc;

/// Types of sensors for a [rover](Rover).
pub enum Sensor {
    /// A sensor that uses lidar.
    Lidar {
        /// The type of lidar, used to determine how to aggregate individual entities.
        ltype: LidarType,
        /// Resolution; bins results into sectors.
        ///
        /// This value represents an angle, in degrees.
        res: f64,
        /// The distance the sensor can "see".
        range: f64,
        /// The position of the sensor.
        ///
        /// When initializing this type of sensor as part of an agent, this field can be any
        /// value. It will be set to the position of the agent.
        pos: Point,
    },
}

impl Sensor {
    /// Returns how far out the sensor can sense.
    pub fn radius(&self) -> f64 {
        match self {
            Sensor::Lidar { range: x, .. } => *x,
        }
    }

    /// Sets the position of the sensor.
    ///
    /// Needed because the sensor doesn't know which rover owns it. Could theoretically have
    /// one of these without a rover attached.
    pub fn set_pos(&mut self, x: f64, y: f64) {
        *self = match &self {
            Sensor::Lidar {
                ltype, res, range, ..
            } => Sensor::Lidar {
                ltype: *ltype,
                res: *res,
                range: *range,
                pos: Point::new(x, y),
            },
        }
    }

    /// Determines which sector a provided point is in.
    fn unbounded_sector(&self, point: Point) -> usize {
        match self {
            Sensor::Lidar { res, pos, .. } => {
                let offset = point - pos;
                let angle = match offset.y.atan2(offset.x).to_degrees() {
                    x if x < 0.0 => x + 360.0,
                    x => x,
                };
                (angle / res).floor() as usize
            }
        }
    }

    /// Retrieves the rewards from the provided agents.
    fn sector_results<T: Agent>(&self, agents: Vec<Rc<T>>) -> Vec<f64> {
        match self {
            Sensor::Lidar {
                ltype,
                res,
                range,
                pos,
            } => {
                let num_sectors = (360.0 / res).ceil() as usize;
                let mut results = vec![Vec::new(); num_sectors];
                for agent in agents {
                    if agent.hidden() {
                        continue;
                    }
                    let dist_sq = na::distance_squared(pos, &agent.pos());
                    if dist_sq > range.powi(2) {
                        continue;
                    }
                    let sector = self.unbounded_sector(agent.pos());
                    results[sector].push(agent.value() / dist_sq.max(0.001));
                }
                results
                    .into_iter()
                    .map(|v| ltype.stat(v).unwrap_or(-1.0))
                    .collect()
            }
        }
    }

    /// Retrieves the [rewards](Agent::reward()) from the provided rovers and [POIs](Poi).
    pub fn scan(&self, rovers: Vec<Rc<Rover>>, pois: Vec<Rc<Poi>>) -> State {
        match self {
            Sensor::Lidar { .. } => {
                let poi_vals = self.sector_results(pois);
                let rover_vals = self.sector_results(rovers);
                let state_len = poi_vals.len() + rover_vals.len();
                let vals = rover_vals.into_iter().chain(poi_vals.into_iter());

                State::from_iterator(state_len, vals)
            }
        }
    }
}

/// Types of [lidar](Sensor) that can be used.
///
/// Determines how [rewards](Agent::reward()) are compiled into a single statistic.
#[derive(Clone, Copy)]
pub enum LidarType {
    /// Aggregates rewards into their average.
    Density,
    /// Aggregates rewards into their maximum.
    Closest,
}

impl LidarType {
    /// Aggregates provided rewards into a single statistic.
    fn stat(&self, items: Vec<f64>) -> Option<f64> {
        let items = na::MatrixXx1::from_vec(items);
        match (items.is_empty(), self) {
            (true, _) => None,
            (_, LidarType::Density) => Some(items.mean()),
            (_, LidarType::Closest) => Some(items.max()),
        }
    }
}

/// Types of constraints an [agent](Agent) may have to [reward](Agent::reward()).
pub enum Constraint {
    /// Constrains by requiring a minimum number of agents watching at the given moment.
    Count(usize),
}
