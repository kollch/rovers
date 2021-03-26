use std::{
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

use nalgebra as na;
use statrs::statistics::Mean;

fn main() {
    println!("Hello, world!");

    let mut rovers = Vec::new();
    // Create agent with lidar, discrete action space and difference reward
    rovers.push(Rc::new(Rover::new(
        DifferenceReward,
        Some(Lidar::new(LidarType::Closest, 90.0, 2.0)),
    )));
    let mut pois = Vec::new();
    pois.push(Rc::new(Poi::new(
        Point::new(2.0, 2.0),
        1.0,
        1.0,
        Constraint::Count(3),
    )));
}

type Point = na::Point2<f64>;
type Vector = na::Vector2<f64>;
type State = na::MatrixXx1<f64>;

static ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

struct Environment<T: EnvInit, S: Sensor, R: Reward> {
    init_policy: T,
    rovers: Vec<Rc<Rover<S, R>>>,
    pois: Vec<Rc<Poi>>,
    size: (f64, f64),
}

impl<T: EnvInit, S: Sensor, R: Reward> Environment<T, S, R> {
    fn reset(&mut self) {
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
                .hid = false;
        }
        // Initialize
        self.init_policy.init(
            self.rovers.iter_mut().collect(),
            self.pois.iter_mut().collect(),
        );
    }

    fn clamp(&mut self) {
        for mut rover in self.rovers.iter_mut() {
            let clamped = na::clamp(
                rover.position,
                Point::origin(),
                Point::new(self.size.0, self.size.1),
            );
            Rc::get_mut(&mut rover)
                .expect("Error: More than one reference to rover.")
                .set_pos(clamped.x, clamped.y);
        }
    }

    fn status(&mut self) -> (Vec<State>, Vec<f64>) {
        let mut states = Vec::new();
        let mut rewards = Vec::new();
        // Observations and rewards
        for rover in self.rovers.iter() {
            states.push(rover.scan(self.rovers.clone(), self.pois.clone()));
            rewards.push(rover.reward(self.rovers.clone(), self.pois.clone()));
        }
        (states, rewards)
    }
}

struct EnvRand;

impl EnvInit for EnvRand {
    fn init_rovers<T: Sensor, R: Reward>(&self, rovers: Vec<&mut Rc<Rover<T, R>>>) {
        for mut rover in rovers {
            Rc::get_mut(&mut rover)
                .expect("Error: More than one reference to rover.")
                .position = Point::origin();
        }
    }

    fn init_pois(&self, pois: Vec<&mut Rc<Poi>>) {
        for mut poi in pois {
            Rc::get_mut(&mut poi)
                .expect("Error: More than one reference to POI.")
                .position = Point::origin();
        }
    }
}

struct EnvCorners {
    span: f64,
}

impl EnvInit for EnvCorners {
    fn init_rovers<T: Sensor, R: Reward>(&self, rovers: Vec<&mut Rc<Rover<T, R>>>) {
        let start = 1.0;
        let end = self.span - 1.0;
        let rad = self.span / f64::sqrt(3.0) / 2.0;
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
                .position = Point::new(x, y);
        }
    }

    fn init_pois(&self, pois: Vec<&mut Rc<Poi>>) {
        let start = 0.0;
        let end = self.span - 1.0;
        for (i, mut poi) in pois.into_iter().enumerate() {
            let offset = i as f64 / 4.0;
            let (x, y) = match i % 4 {
                0 => (start + offset, start + offset),
                1 => (end - offset, start + offset),
                2 => (start + offset, end - offset),
                3 => (end - offset, end - offset),
                _ => unreachable!(),
            };
            Rc::get_mut(&mut poi)
                .expect("Error: More than one reference to POI.")
                .position = Point::new(x, y);
        }
    }
}

struct Rover<T: Sensor, R: Reward> {
    ident: usize,
    position: Point,
    path: Vec<Point>,
    reward_type: R,
    sensor: Option<T>,
}

impl<T: Sensor, R: Reward> Rover<T, R> {
    fn new(reward_type: R, sensor: Option<T>) -> Self {
        Rover {
            ident: ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            position: Point::origin(),
            path: Vec::new(),
            reward_type,
            sensor,
        }
    }

    fn scan(&self, rovers: Vec<Rc<Rover<T, R>>>, pois: Vec<Rc<Poi>>) -> State {
        // Remove self from the list of rovers
        let rovers = self.without_self(rovers);

        self.sensor
            .as_ref()
            .map_or(State::from_element(1, -1.0), |s| s.scan(rovers, pois))
    }

    // TODO: How would you handle the case where a POI only gives out its reward once? If two
    // rovers access it at the same time (in the same frame), which one does the reward go to? It
    // could depend on where the rover is in the array of rovers.
    fn reward(&self, rovers: Vec<Rc<Rover<T, R>>>, pois: Vec<Rc<Poi>>) -> f64 {
        self.reward_type.calculate(self.ident, rovers, pois)
    }

    fn reset(&mut self) {
        self.set_pos(0.0, 0.0);
        self.path.clear();
    }
}

impl<T: Sensor> Default for Rover<T, DefaultReward> {
    fn default() -> Self {
        Self::new(DefaultReward, None)
    }
}

impl<T: Sensor, R: Reward> Agent for Rover<T, R> {
    fn id(&self) -> usize {
        self.ident
    }

    fn pos(&self) -> Point {
        self.position
    }

    fn set_pos(&mut self, x: f64, y: f64) {
        self.position = Point::new(x, y);
        self.sensor.as_mut().map(|s| s.set_pos(x, y));
        self.path.push(self.position);
    }

    fn radius(&self) -> f64 {
        self.sensor.as_ref().map_or(1.0, |s| s.radius())
    }

    fn value(&self) -> f64 {
        1.0
    }

    fn hidden(&self) -> bool {
        false
    }
}

struct DefaultReward;

impl Reward for DefaultReward {
    fn calculate<T: Sensor, R: Reward>(
        &self,
        _id: usize,
        rovers: Vec<Rc<Rover<T, R>>>,
        pois: Vec<Rc<Poi>>,
    ) -> f64 {
        pois.into_iter()
            .map(|poi| poi.give_reward(&rovers).unwrap_or(0.0))
            .sum()
    }
}

struct DifferenceReward;

impl Reward for DifferenceReward {
    fn calculate<T: Sensor, R: Reward>(
        &self,
        id: usize,
        rovers: Vec<Rc<Rover<T, R>>>,
        pois: Vec<Rc<Poi>>,
    ) -> f64 {
        let reward = DefaultReward.calculate(id, rovers.clone(), pois.clone());
        let rovers = without_id(id, rovers);
        let reward_without_self = DefaultReward.calculate(id, rovers, pois);
        reward - reward_without_self
    }
}

struct Lidar {
    ltype: LidarType,
    res: f64,
    range: f64,
    position: Point,
}

impl Lidar {
    fn new(ltype: LidarType, resolution: f64, range: f64) -> Self {
        Lidar {
            ltype,
            res: resolution,
            range,
            position: Point::origin(),
        }
    }

    fn unbounded_sector(&self, point: Point) -> usize {
        let offset = point - self.position;
        let angle = match offset.y.atan2(offset.x).to_degrees() {
            x if x < 0.0 => x + 360.0,
            x => x,
        };
        (angle / self.res).floor() as usize
    }

    fn sector(&self, point: Point) -> Option<usize> {
        if na::distance(&self.position, &point) > self.range {
            None
        } else {
            Some(self.unbounded_sector(point))
        }
    }

    fn sector_results<T: Agent>(&self, agents: Vec<Rc<T>>) -> Vec<f64> {
        let num_sectors = (360.0 / self.res).ceil() as usize;
        let mut results = vec![Vec::new(); num_sectors];
        for agent in agents {
            if agent.hidden() {
                continue;
            }
            let dist_sq = na::distance_squared(&self.position, &agent.pos());
            if dist_sq > self.range.powi(2) {
                continue;
            }
            let sector = self.unbounded_sector(agent.pos());
            results[sector].push(agent.value() / dist_sq.max(0.001));
        }
        results
            .into_iter()
            .map(|v| self.ltype.stat(&v).unwrap_or(-1.0))
            .collect()
    }
}

impl Sensor for Lidar {
    fn radius(&self) -> f64 {
        self.range
    }

    fn set_pos(&mut self, x: f64, y: f64) {
        self.position = Point::new(x, y);
    }

    fn scan<T: Sensor, R: Reward>(
        &self,
        rovers: Vec<Rc<Rover<T, R>>>,
        pois: Vec<Rc<Poi>>,
    ) -> State {
        let poi_vals = self.sector_results(pois);
        let rover_vals = self.sector_results(rovers);

        State::from_iterator(1, rover_vals.into_iter().chain(poi_vals.into_iter()))
    }
}

enum LidarType {
    Density,
    Closest,
}

impl LidarType {
    fn stat(&self, items: &[f64]) -> Option<f64> {
        match (items.is_empty(), self) {
            (true, _) => None,
            (_, LidarType::Density) => Some(items.mean()),
            (_, LidarType::Closest) => {
                Some(items.into_iter().fold(f64::NEG_INFINITY, |a, &e| a.max(e)))
            }
        }
    }
}

enum Constraint {
    Count(usize),
}

struct Poi {
    ident: usize,
    position: Point,
    obs_radius: f64,
    val: f64,
    hid: bool,
    constraint: Constraint,
}

impl Poi {
    fn new(position: Point, obs_radius: f64, val: f64, constraint: Constraint) -> Self {
        Poi {
            ident: ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            position,
            obs_radius,
            val,
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

    fn hidden(&self) -> bool {
        self.hid
    }
}

impl Rewarder for Poi {
    fn give_reward<T: Agent>(&self, agents: &[Rc<T>]) -> Option<f64> {
        match (self.hid, &self.constraint) {
            (false, Constraint::Count(x)) if self.num_observing(agents) >= *x => Some(self.val),
            (_, Constraint::Count(_)) => None,
        }
    }

    fn num_observing<T: Agent>(&self, agents: &[Rc<T>]) -> usize {
        agents
            .iter()
            .filter(|a| {
                let dist = na::distance(&a.pos(), &self.position);
                dist <= self.obs_radius && dist <= a.radius()
            })
            .count()
    }
}

fn without_id<A: Agent>(id: usize, agents: Vec<Rc<A>>) -> Vec<Rc<A>> {
    agents
        .into_iter()
        .filter_map(|a| (a.id() != id).then(|| Rc::clone(&a)))
        .collect()
}

trait EnvInit {
    fn init_rovers<T: Sensor, R: Reward>(&self, rovers: Vec<&mut Rc<Rover<T, R>>>);
    fn init_pois(&self, pois: Vec<&mut Rc<Poi>>);
    fn init<T: Sensor, R: Reward>(
        &self,
        rovers: Vec<&mut Rc<Rover<T, R>>>,
        pois: Vec<&mut Rc<Poi>>,
    ) {
        self.init_rovers(rovers);
        self.init_pois(pois);
    }
}

trait Agent {
    fn id(&self) -> usize;
    fn pos(&self) -> Point;
    fn set_pos(&mut self, x: f64, y: f64);
    fn radius(&self) -> f64;
    fn value(&self) -> f64;
    fn hidden(&self) -> bool;
    fn update_pos(&mut self, v: Vector) {
        let new_pos = self.pos() + v;
        self.set_pos(new_pos.x, new_pos.y);
    }
    fn without_self<A: Agent>(&self, agents: Vec<Rc<A>>) -> Vec<Rc<A>> {
        without_id(self.id(), agents)
    }
}

trait Rewarder {
    fn give_reward<T: Agent>(&self, agents: &[Rc<T>]) -> Option<f64>;
    fn num_observing<T: Agent>(&self, agents: &[Rc<T>]) -> usize;
}

trait Sensor {
    fn radius(&self) -> f64;
    fn set_pos(&mut self, x: f64, y: f64);
    fn scan<T: Sensor, R: Reward>(&self, rovers: Vec<Rc<Rover<T, R>>>, pois: Vec<Rc<Poi>>)
        -> State;
}

trait Reward {
    fn calculate<T: Sensor, R: Reward>(
        &self,
        id: usize,
        rovers: Vec<Rc<Rover<T, R>>>,
        pois: Vec<Rc<Poi>>,
    ) -> f64;
}
