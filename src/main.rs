use std::{
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

use nalgebra as na;
use rand::prelude::*;

fn main() {
    println!("Hello, world!");

    // Create some rovers
    let mut rovers = Vec::new();
    rovers.push(Rc::new(Rover::new(
        Reward::Difference,
        Some(Sensor::Lidar {
            ltype: LidarType::Closest,
            res: 45.0,
            range: 2.0,
            pos: Point::origin(),
        }),
    )));
    rovers.push(Rc::new(Rover::new(
        Reward::Default,
        Some(Sensor::Lidar {
            ltype: LidarType::Density,
            res: 90.0,
            range: 1.0,
            pos: Point::origin(),
        }),
    )));
    rovers.push(Rc::new(Rover::new(
        Reward::Default,
        Some(Sensor::Lidar {
            ltype: LidarType::Density,
            res: 90.0,
            range: 3.0,
            pos: Point::origin(),
        }),
    )));

    // Create some POIs
    let mut pois = Vec::new();
    pois.push(Rc::new(Poi::new(
        Point::origin(),
        1.0,
        1.0,
        Constraint::Count(3),
    )));
    pois.push(Rc::new(Poi::new(
        Point::origin(),
        1.0,
        1.0,
        Constraint::Count(2),
    )));
    pois.push(Rc::new(Poi::new(
        Point::origin(),
        1.0,
        1.0,
        Constraint::Count(5),
    )));

    // Create an environment
    let mut env = Environment::new(EnvCorners { span: 10.0 }, rovers, pois, (10.0, 10.0));
    env.reset();

    let mut actions: Vec<Vector> = Vec::new();
    for _ in 0..env.rovers.len() {
        actions.push(Vector::new(random(), random()));
    }

    let (states, _rewards) = env.step(actions);

    println!("Sample environment state (each row corresponds to the state of a rover):");
    for state in states {
        println!("{}", state);
    }
}

type Point = na::Point2<f64>;
type Vector = na::Vector2<f64>;
type State = na::MatrixXx1<f64>;

static ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

struct Environment<T: EnvInit> {
    init_policy: T,
    rovers: Vec<Rc<Rover>>,
    pois: Vec<Rc<Poi>>,
    size: (f64, f64),
}

impl<T: EnvInit> Environment<T> {
    fn new(init_policy: T, rovers: Vec<Rc<Rover>>, pois: Vec<Rc<Poi>>, size: (f64, f64)) -> Self {
        Environment {
            init_policy,
            rovers,
            pois,
            size,
        }
    }

    fn step(&mut self, actions: Vec<Vector>) -> (Vec<State>, Vec<f64>) {
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

    fn reset(&mut self) -> (Vec<State>, Vec<f64>) {
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
        self.status()
    }

    fn clamp_positions(&mut self) {
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

    fn status(&self) -> (Vec<State>, Vec<f64>) {
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
    fn init_rovers(&self, rovers: Vec<&mut Rc<Rover>>) {
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
    fn init_rovers(&self, rovers: Vec<&mut Rc<Rover>>) {
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

struct Rover {
    ident: usize,
    position: Point,
    path: Vec<Point>,
    reward_type: Reward,
    sensor: Option<Sensor>,
}

impl Rover {
    fn new(reward_type: Reward, sensor: Option<Sensor>) -> Self {
        Rover {
            ident: ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            position: Point::origin(),
            path: Vec::new(),
            reward_type,
            sensor,
        }
    }

    fn scan(&self, rovers: Vec<Rc<Rover>>, pois: Vec<Rc<Poi>>) -> State {
        // Remove self from the list of rovers
        let rovers = self.without_self(rovers);

        self.sensor
            .as_ref()
            .map_or(State::from_element(1, -1.0), |s| s.scan(rovers, pois))
    }

    // TODO: How would you handle the case where a POI only gives out its reward once? If two
    // rovers access it at the same time (in the same frame), which one does the reward go to? It
    // could depend on where the rover is in the array of rovers.
    fn reward(&self, rovers: Vec<Rc<Rover>>, pois: Vec<Rc<Poi>>) -> f64 {
        self.reward_type.calculate(self.ident, rovers, pois)
    }

    fn reset(&mut self) {
        self.set_pos(0.0, 0.0);
        self.path.clear();
    }

    fn act(&mut self, action: Vector) {
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

    fn hidden(&self) -> bool {
        false
    }
}

enum Reward {
    Default,
    Difference,
}

impl Reward {
    fn calculate(&self, id: usize, rovers: Vec<Rc<Rover>>, pois: Vec<Rc<Poi>>) -> f64 {
        let default_calc = |rovers: Vec<Rc<Rover>>, pois: Vec<Rc<Poi>>| {
            pois.into_iter()
                .map(|poi| poi.give_reward(&rovers).unwrap_or(0.0))
                .sum()
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

enum Sensor {
    Lidar {
        ltype: LidarType,
        res: f64,
        range: f64,
        pos: Point,
    },
}

impl Sensor {
    fn radius(&self) -> f64 {
        match self {
            Sensor::Lidar { range: x, .. } => *x,
        }
    }

    fn set_pos(&mut self, x: f64, y: f64) {
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

    fn scan(&self, rovers: Vec<Rc<Rover>>, pois: Vec<Rc<Poi>>) -> State {
        match self {
            Sensor::Lidar { .. } => {
                let poi_vals = self.sector_results(pois);
                let rover_vals = self.sector_results(rovers);

                State::from_iterator(1, rover_vals.into_iter().chain(poi_vals.into_iter()))
            }
        }
    }
}

#[derive(Clone, Copy)]
enum LidarType {
    Density,
    Closest,
}

impl LidarType {
    fn stat(&self, items: Vec<f64>) -> Option<f64> {
        let items = na::MatrixXx1::from_vec(items);
        match (items.is_empty(), self) {
            (true, _) => None,
            (_, LidarType::Density) => Some(items.mean()),
            (_, LidarType::Closest) => Some(items.max()),
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
    fn init_rovers(&self, rovers: Vec<&mut Rc<Rover>>);
    fn init_pois(&self, pois: Vec<&mut Rc<Poi>>);
    fn init(&self, rovers: Vec<&mut Rc<Rover>>, pois: Vec<&mut Rc<Poi>>) {
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
