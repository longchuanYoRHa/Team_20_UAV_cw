import numpy as np
import random

class Wind:
    def __init__(self, max_steady_state=0.1, max_gust=0.2, k_gusts=0.1):
        self.max_steady_state = max_steady_state
        self.max_gust = max_gust
        self.k_gusts = k_gusts
        self.t = 0
        self.gust_params = []
        self.last_gust_t0 = 0
        
        self.master_angle = random.uniform(0, 2 * np.pi)
        self.master_speed = random.uniform(0, self.max_steady_state)
        
        self.current_wind = np.array([0.0, 0.0, 0.0])
        self.calc_init_wind()

    def calc_init_wind(self):
        self.current_wind = np.array([
            self.master_speed * np.cos(self.master_angle),
            self.master_speed * np.sin(self.master_angle),
            0.0
        ])

    def step(self, dt):
        self.t += dt
        
        variation_angle = np.sin(self.t * 0.5) * 0.15  
        variation_speed = np.cos(self.t * 0.7) * (self.master_speed * 0.1) 
        
        actual_angle = self.master_angle + variation_angle
        actual_speed = self.master_speed + variation_speed
        
        steady_wind = np.array([
            actual_speed * np.cos(actual_angle),
            actual_speed * np.sin(actual_angle),
            0.0
        ])

        current_gust = np.array([0.0, 0.0, 0.0])
        
        if (self.t - self.last_gust_t0) > random.uniform(0, 1 / (self.k_gusts + 0.1)):
            theta = self.master_angle + random.uniform(-np.pi/4, np.pi/4) 
            wg0 = random.uniform(0, self.max_gust)
            lg = np.exp(random.uniform(np.log(0.1), np.log(2.0))) 
            self.gust_params.append({'theta': theta, 'wg0': wg0, 'lg': lg, 't0': self.t})
            self.last_gust_t0 = self.t

        active_gusts = []
        for g in self.gust_params:
            rel_t = self.t - g['t0']
            if rel_t < g['lg']:
                gust_v = (g['wg0'] / 2.0) * (1 - np.cos((2 * np.pi * rel_t) / g['lg']))
                current_gust += np.array([
                    np.cos(g['theta']) * gust_v, 
                    np.sin(g['theta']) * gust_v, 
                    0.0
                ])
                active_gusts.append(g)
        self.gust_params = active_gusts
        
        return steady_wind + current_gust

    def get_wind(self, dt):
        return self.step(dt)