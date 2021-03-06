use std::rc::Rc;

use rand::prelude::*;

use rovers::{
    agents::{Poi, Rover},
    env::{EnvCorners, Environment},
    sensors::{Constraint, LidarType, Sensor},
    Point, Reward, Vector,
};

fn main() {
    println!("Hello, world!");

    // Create some rovers
    let rovers = vec![
        Rc::new(Rover::new(
            Reward::Difference,
            Some(Sensor::Lidar {
                ltype: LidarType::Closest,
                res: 45.0,
                range: 2.0,
                pos: Point::origin(),
            }),
        )),
        Rc::new(Rover::new(
            Reward::Default,
            Some(Sensor::Lidar {
                ltype: LidarType::Density,
                res: 90.0,
                range: 1.0,
                pos: Point::origin(),
            }),
        )),
        Rc::new(Rover::new(
            Reward::Default,
            Some(Sensor::Lidar {
                ltype: LidarType::Density,
                res: 90.0,
                range: 3.0,
                pos: Point::origin(),
            }),
        )),
    ];

    // Create some POIs
    let pois = vec![
        Rc::new(Poi::new(Point::origin(), 3.0, 1.0, Constraint::Count(3))),
        Rc::new(Poi::new(Point::origin(), 2.0, 1.0, Constraint::Count(3))),
        Rc::new(Poi::new(Point::origin(), 5.0, 1.0, Constraint::Count(3))),
    ];

    // Create an environment
    let mut env = Environment::new(EnvCorners, rovers, pois, (10.0, 10.0));
    env.reset();

    let mut actions: Vec<Vector> = Vec::new();
    for _ in 0..env.num_rovers {
        actions.push(Vector::new(random(), random()));
    }
    // Use this for the moment as a way to have a hardcoded seed - now it matches the results of
    // the original project
    /*
    let actions = vec![
        Vector::new(0.680375, -0.211234),
        Vector::new(0.566198, 0.59688),
        Vector::new(0.823295, -0.604897),
        Vector::new(-0.329554, 0.536459),
    ];
    */

    let (states, _rewards) = env.step(actions);

    println!("Sample environment state (each row corresponds to the state of a rover):");
    for state in states {
        println!("{:?}", state.data.as_vec());
    }
}
