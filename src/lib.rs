//! A prototype rover environment for reinforcement learning.

#![warn(missing_docs)]

use std::rc::Rc;

use nalgebra as na;

/// A 2D point.
pub type Point = na::Point2<f64>;
/// A 2D vector.
pub type Vector = na::Vector2<f64>;
type State = na::MatrixXx1<f64>;

/// Environment utilities to provide bounds on the world the agents interact in.
pub mod env {
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
}

/// Agents of the simulation.
pub mod agents {
    use super::{
        sensors::{Constraint, Sensor},
        Point, Reward, State, Vector,
    };
    use nalgebra as na;
    use std::{
        rc::Rc,
        sync::atomic::{AtomicUsize, Ordering},
    };

    static ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

    /// A rover; contains an [optional](Option) [sensor](Sensor).
    pub struct Rover {
        ident: usize,
        position: Point,
        path: Vec<Point>,
        reward_type: Reward,
        sensor: Option<Sensor>,
    }

    impl Rover {
        /// Constructs a new `Rover` located at the origin with an empty path and unique ID.
        pub fn new(reward_type: Reward, sensor: Option<Sensor>) -> Self {
            Rover {
                ident: ID_COUNTER.fetch_add(1, Ordering::SeqCst),
                position: Point::origin(),
                path: Vec::new(),
                reward_type,
                sensor,
            }
        }

        /// Returns a [column vector][state] of the [rover](Rover) based on the rovers and
        /// [POIs](Poi) around it.
        ///
        /// This column vector is generated from the state determined by the rover's sensor. If
        /// there is no sensor on the rover, it returns a 1x1 column vector with its singular
        /// element -1.
        ///
        /// [state]: na::MatrixXx1<f64>
        pub fn scan(&self, rovers: Vec<Rc<Rover>>, pois: Vec<Rc<Poi>>) -> State {
            // Remove self from the list of rovers
            let rovers = self.without_self(rovers);

            self.sensor
                .as_ref()
                .map_or(State::from_element(1, -1.0), |s| s.scan(rovers, pois))
        }

        /// Determines the value rewarded from rovers and [POIs](Poi) around the rover.
        ///
        /// TODO: How would you handle the case where a POI only gives out its reward once? If two
        /// rovers access it at the same time (in the same frame), which one does the reward go to?
        /// It could depend on where the rover is in the array of rovers.
        pub fn get_reward(&self, rovers: Vec<Rc<Rover>>, pois: Vec<Rc<Poi>>) -> f64 {
            self.reward_type.calculate(self.ident, rovers, pois)
        }

        /// Moves the rover based on a specified action.
        pub fn act(&mut self, action: Vector) {
            let curr = self.position;
            self.set_pos(curr.x + action.x, curr.y + action.y);
        }
    }

    impl Default for Rover {
        fn default() -> Self {
            Self::new(Reward::Default, None)
        }
    }

    impl Agent for Rover {
        fn id(&self) -> usize {
            self.ident
        }

        fn pos(&self) -> Point {
            self.position
        }

        fn set_pos(&mut self, x: f64, y: f64) {
            self.position = Point::new(x, y);
            if let Some(s) = self.sensor.as_mut() {
                s.set_pos(x, y);
            }
            self.path.push(self.position);
        }

        fn radius(&self) -> f64 {
            self.sensor.as_ref().map_or(1.0, |s| s.radius())
        }

        fn value(&self) -> f64 {
            1.0
        }

        fn reward<T: Agent>(&self, _agents: &[Rc<T>]) -> f64 {
            self.value()
        }

        fn hidden(&self) -> bool {
            false
        }

        /// Resets the rover by clearing its path and setting its position to (0, 0).
        fn reset(&mut self) {
            self.set_pos(0.0, 0.0);
            self.path.clear();
        }
    }

    /// A POI; gives a [reward](Reward) based on its [constraint](Constraint).
    pub struct Poi {
        ident: usize,
        position: Point,
        val: f64,
        obs_radius: f64,
        hid: bool,
        constraint: Constraint,
    }

    impl Poi {
        /// Constructs a new POI with a unique ID.
        pub fn new(position: Point, val: f64, obs_radius: f64, constraint: Constraint) -> Self {
            Poi {
                ident: ID_COUNTER.fetch_add(1, Ordering::SeqCst),
                position,
                val,
                obs_radius,
                hid: false,
                constraint,
            }
        }
    }

    impl Agent for Poi {
        fn id(&self) -> usize {
            self.ident
        }

        fn pos(&self) -> Point {
            self.position
        }

        fn set_pos(&mut self, x: f64, y: f64) {
            self.position = Point::new(x, y);
        }

        fn radius(&self) -> f64 {
            self.obs_radius
        }

        fn value(&self) -> f64 {
            self.val
        }

        fn reward<T: Agent>(&self, agents: &[Rc<T>]) -> f64 {
            match (self.hid, &self.constraint) {
                (false, Constraint::Count(x)) if self.num_observing(agents) >= *x => self.val,
                (_, Constraint::Count(_)) => 0.0,
            }
        }

        fn hidden(&self) -> bool {
            self.hid
        }

        /// Resets the POI by making it not hidden.
        fn reset(&mut self) {
            self.hid = false;
        }
    }

    /// Provides basic attributes and functionality of an agent.
    pub trait Agent {
        /// Provides the ID of the agent.
        fn id(&self) -> usize;
        /// Provides the position of the agent.
        fn pos(&self) -> Point;
        /// Sets the position of the agent.
        fn set_pos(&mut self, x: f64, y: f64);
        /// Provides the distance the agent can 'see' around itself.
        fn radius(&self) -> f64;
        /// The value of a possible reward.
        ///
        /// This is meant to be used only if all constraints for giving a reward are met.
        fn value(&self) -> f64;
        /// Returns a reward, provided the agent met its constraints.
        fn reward<T: Agent>(&self, agents: &[Rc<T>]) -> f64;
        /// Checks if the agent can be seen by other agents, even if within range.
        ///
        /// Useful to intentionally prevent reward-giving.
        fn hidden(&self) -> bool;
        /// Resets the agent.
        fn reset(&mut self);
        /// Updates the position of the agent based on a provided offset.
        fn update_pos(&mut self, v: Vector) {
            let new_pos = self.pos() + v;
            self.set_pos(new_pos.x, new_pos.y);
        }
        /// Filters the provided agents to remove itself from the `Vec`.
        fn without_self<A: Agent>(&self, agents: Vec<Rc<A>>) -> Vec<Rc<A>> {
            without_id(self.id(), agents)
        }
        /// Returns the number of agents observing the agent.
        fn num_observing<T: Agent>(&self, agents: &[Rc<T>]) -> usize {
            if self.hidden() {
                0
            } else {
                agents
                    .iter()
                    .filter(|a| {
                        let dist = na::distance(&a.pos(), &self.pos());
                        dist <= self.radius() && dist <= a.radius()
                    })
                    .count()
            }
        }
    }

    /// Filters a [`Vec`] of [agents](Agent), ignoring any that match the provided ID.
    pub fn without_id<A: Agent>(id: usize, agents: Vec<Rc<A>>) -> Vec<Rc<A>> {
        agents
            .into_iter()
            .filter_map(|a| (a.id() != id).then(|| Rc::clone(&a)))
            .collect()
    }
}

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

/// Sensors used by [rovers](agents::Rover).
pub mod sensors {
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
}
