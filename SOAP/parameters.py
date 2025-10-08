import numpy as np

from better_uniform import buniform

def get_all_parameters(sim):
    star_pars = ['prot', 'incl', 'u1', 'u2']
    ar_template_pars = ['lat', 'long', 'size']
    planet_pars = ['a', 'e', 'ip', 'lbda', 't0', 'w', 'Rp', 'Pp']
    ring_pars = ['fi', 'fe', 'ir', 'theta']

    r = {'star': star_pars}

    if len(sim.active_regions) > 0:
        ar_pars = []
        for i,ar in enumerate(sim.active_regions):
            ar_pars.append(f'lat{i+1}')
            ar_pars.append(f'long{i+1}')
            ar_pars.append(f'size{i+1}')
        
        r.update({'ar': ar_pars})


    if sim.has_planet: # has planet
        if sim.planet.ring: # planet has ring
            r.update({'planet':planet_pars, 'ring': ring_pars})
        else:
            r.update({'planet':planet_pars})
    
    return r

def merge_all_parameters(possible):
    merged = []
    for p in possible:
        for par in possible[p]:
            merged.append(p + '.' + par)
    return merged

def get_initial(sim, vary):
    p0 = []
    for v in vary:
        _1, _2 = v.split('.')
        if _1 == 'ar':
            i = int(_2[-1]) - 1
            _2 = _2[:-1]
            p0.append(getattr(getattr(sim, 'active_regions')[i], _2))
        elif _1 == 'ring':
            p0.append(getattr(getattr(sim.planet, _1), _2))
        else:
            p0.append(getattr(getattr(sim, _1), _2))
    
    return p0


def get_bounds(sim, vary):
    defaults = {
        'star.prot': (1, 200),
        'star.incl': (-90, 90),
        'star.u1': (0, 1),
        'star.u2': (0, 1),
        # 
        'ar.lat': (-90, 90),
        'ar.long': (-180, 180),
        'ar.size': (0, .5),
        # 
        'planet.Pp': (0.1, 2000),
        'planet.t0': None,
        'planet.e': (0.0, 1.0),
        'planet.w': (0.0, 360.0),
        'planet.ip': (0.0, 90.0),
        'planet.lbda': (0.0, 360.0),
        'planet.a': (1, 400),
        'planet.Rp': (0.0, 0.15),
        # 
        'ring.fi': (1.0, 10),
        'ring.fe': (1.0, 10),
        'ring.ir': (0.0, 90.0),
        'ring.theta': (0.0, 180.0),
    }
    
    bounds = []

    for v in vary:
        if v in defaults:
            bounds.append(defaults[v])
        elif v[:-1] in defaults:
            bounds.append(defaults[v[:-1]])
        else:
            bounds.append(None)

    return bounds


def get_priors(sim, vary, prior_form=None):
    priors = []
    if prior_form is None:
        bounds = get_bounds(sim, vary)
        bounds = [list(b) for b in bounds]
        for b in bounds:
            if b[0] is None: b[0] = -np.inf
            if b[1] is None: b[1] = np.inf
            priors.append(buniform(*b))
    else:
        raise NotImplementedError

    return priors


def get_prior_width(sim, vary):
    w = []
    priors = get_priors(sim, vary)
    for p in priors:
        support = p.interval(1)
        w.append(np.ediff1d(support)[0])
    return w

def get_random_from_prior(sim, vary):
    r = []
    priors = get_priors(sim, vary)
    for p in priors:
        r.append(p.rvs())
    return r


def check_inside_bounds(sim, pars, bounds):
    for p, b in zip(pars, bounds):
        if b[0] <= p and p <= b[1]:
            pass
        else:
            return False
    return True


def update_values(sim, vary, new_values):
    for v, newv in zip(vary, new_values):
        _1, _2 = v.split('.')
        if _1 == 'ar':
            i = int(_2[-1]) - 1
            _2 = _2[:-1]
            setattr(getattr(sim, 'active_regions')[i], _2, newv)
        elif _1 == 'ring':
            setattr(getattr(sim.planet, _1), _2, newv)
        else:
            setattr(getattr(sim, _1), _2, newv)