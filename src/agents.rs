//! Agents of the simulation.

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
