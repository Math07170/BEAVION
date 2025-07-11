#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:22:21 2019

@author: DRUOT Thierry : original Scilab implementation
         PETEILH Nicolas : portage to Python
"""

import numpy

from copy import deepcopy


def kg_t(t): return t*1000.
def t_kg(kg): return kg/1000.

def kg_T(T): return T*1000.
def T_kg(kg): return kg/1000.

def year_s(s): return s/31557600.
def s_year(year): return year*31557600.

def km2_m2(m2): return m2*1.e-6
def m2_km2(km2): return km2*1.e6

def kW_W(W): return W*1.e-3
def W_kW(kW): return kW*1.e3

def MW_W(W): return W*1.e-6
def W_MW(MW): return MW*1.e6

def GW_W(W): return W*1.e-9
def W_GW(GW): return GW*1.e9

def s_min(min): return min*60.  # Translate minutes into seconds
def min_s(s): return s/60.      # Translate seconds into minutes

def s_h(h): return h*3600.   # Translate hours into seconds
def h_s(s): return s/3600.   # Translate seconds into hours

def m_inch(inch): return inch*0.0254   # Translate inch into metres
def inch_m(m): return m/0.0254      # Translate metres into inch

def m_ft(ft): return ft*0.3048   # Translate feet into metres
def ft_m(m): return m/0.3048   # Translate metres into feet

def m_km(km): return km*1000.   # Translate kilometer into metres
def km_m(m): return m/1000.   # Translate metres into kilometer

def m_NM(NM): return NM*1852.   # Translate nautical miles into metres
def NM_m(m): return m/1852.   # Translate metres into nautical miles

def mps_kmph(kmph): return kmph*1000./3600.   # Translate knots into meters per second
def kmph_mps(mps): return mps*3600./1000.   # Translate knots into meters per second

def mps_kt(kt): return kt*1852./3600.   # Translate knots into meters per second
def kt_mps(mps): return mps*3600./1852.   # Translate meters per second into knots

def mps_ftpmin(ftpmin): return ftpmin*0.3048/60.   # Translate feet per minutes into meters per second
def ftpmin_mps(mps): return mps/0.3048*60.   # Translate meters per second into feet per minutes

def liter_usgal(usgal): return usgal*3.7853982   # Translate US gallons into liters
def usgal_liter(liter): return liter/3.7853982   # Translate liters into US gallons

def m3_usgal(usgal): return usgal*0.0037853982   # Translate US gallons into m3
def usgal_m3(m3): return m3/0.0037853982   # Translate m3 into US gallons

def rad_deg(deg): return deg*numpy.pi/180.   # Translate degrees into radians
def deg_rad(rad): return rad*180./numpy.pi   # Translate radians into degrees

def J_MJ(MJ): return MJ*1.e6   # Translate MJ into J
def MJ_J(J): return J*1.e-6   # Translate J into MJ

def J_kJ(kJ): return kJ*1.e3   # Translate kJ into J
def kJ_J(J): return J*1.e-3   # Translate J into kJ

def Wh_kWh(kWh): return kWh*1.e3   # Translate kWh into Wh
def kWh_Wh(Wh): return Wh*1.e-3   # Translate Wh into kWh

def J_Wh(Wh): return Wh*3.6e3   # Translate kWh into J
def Wh_J(J): return J/3.6e3   # Translate J into kWh

def J_kWh(kWh): return kWh*3.6e6   # Translate kWh into J
def kWh_J(J): return J/3.6e6   # Translate J into kWh

def J_MWh(MWh): return MWh*3.6e9   # Translate MWh into J
def MWh_J(J): return J/3.6e9   # Translate J into MWh

def J_GWh(GWh): return GWh*3.6e12   # Translate MWh into J
def GWh_J(J): return J/3.6e12   # Translate J into MWh

def J_TWh(TWh): return TWh*3.6e15   # Translate MWh into J
def TWh_J(J): return J/3.6e15   # Translate J into MWh

def Jpm_kWhpkm(kWhpkm): return kWhpkm*3.6e3   # Translate kWh/km into J/m
def kWhpkm_Jpm(Jpm): return Jpm/3.6e3   # Translate J/m into kWh/km

def daN_N(N): return N/10.   # Translate N into daN
def N_daN(daN): return daN*10.   # Translate daN into N

def N_kN(kN): return kN*1000.   # Translate kN into N
def kN_N(N): return N/1000.   # Translate N into kN

def Pa_mbar(mbar): return mbar*1.e2   # Translate mbar into Pascal
def mbar_Pa(Pa): return Pa/1.e2   # Translate Pascal into mbar

def Pa_bar(bar): return bar*1.e5   # Translate bar into Pascal
def bar_Pa(Pa): return Pa/1.e5   # Translate Pascal into bar

def Pam3pkg_barLpkg(barLpkg): return barLpkg*1.e2  # Translate bar.L/kg into Pa.m3/kg
def barLpkg_Pam3pkg(Pam3pkg): return Pam3pkg/1.e2  # Translate Pa.m3/kg into bar.L/kg

def pc_no_dim(no_dim): return no_dim*100.   # Translate no dimension value into percentile
def no_dim_pc(no_dim): return no_dim/100.   # Translate percentile into no dimension value

def kgpWps_lbpshpph(lbpshpph): return lbpshpph*1.68969e-07  # Ttranslate lb/shp/h into kg/W/s
def lbpshpph_kgpWps(kgpWps_): return kgpWps_/1.68969e-07    # Ttranslate kg/W/s into lb/shp/h

def smart_round(X, S):
    Fac = (10 * numpy.ones(S))**numpy.min(4, max(0, 4 - round(numpy.log10(S))))
    return round(X * Fac)  # Fac


#=========================================================================
#
#	Generic unit converter
#
#=========================================================================
UNIT = {}

# dim = "Distance"
UNIT["m"] = 1.
UNIT["cm"] = 0.01
UNIT["mm"] = 0.001
UNIT["inch"] = 0.0254
UNIT["in"] = 0.0254
UNIT["km"] = 1000.
UNIT["ft"] = 0.3048
UNIT["NM"] = 1852.

# dim = "YearlyDistance"
UNIT["km/year"] = 1.
UNIT["1e12.km/year"] = 1.e-12

# dim = "Area"
UNIT["m2"] = 1.
UNIT["cm2"] = 0.0001
UNIT["ft2"] = 0.0929030
UNIT["inch2"] = 0.00064516
UNIT["in2"] = 0.00064516

# dim = "AreaPerMass"
UNIT["m2/kg"] = 1.
UNIT["cm2/kg"] = 0.0001
UNIT["ft2/kg"] = 0.0929030
UNIT["inch2/kg"] = 0.00064516
UNIT["in2/kg"] = 0.00064516

# dim = "PowerPerMass"
UNIT["N/kg"] = 1.
UNIT["daN/kg"] = 10.
UNIT["kN/kg"] = 1000.

# dim = "Duration"
UNIT["s"] = 1.
UNIT["min"] = 60.
UNIT["h"] = 3600.
UNIT["an"] = 31557600.
UNIT["year"] = 31557600.

# dim = "Velocity"
UNIT["m/s"] = 1.
UNIT["ft/s"] = 0.3048
UNIT["ft/min"] = 0.00508
UNIT["km/h"] = 0.2777777778
UNIT["kt"] = 0.5144444444
UNIT["mph"] = 0.4469444

# dim = "Acceleration"
UNIT["m/s2"] = 1.
UNIT["km/s2"] = 1000.
UNIT["ft/s2"] = 0.3048
UNIT["kt/s"] = 0.5144444444

# dim = "AbsoluteTemperature"
UNIT["KELVIN"] = 1.
UNIT["Kelvin"] = 1.
UNIT["K"] = 1.
UNIT["RANKINE"] = 0.5555556
UNIT["Rankine"] = 0.5555556
UNIT["R"] = 0.5555556

# dim = "AbsoluteTemperatureinCelsius"
UNIT["CELSIUS"] = 1.
UNIT["Celsius"] = 1.
UNIT["C"] = 1.

# dim = "DeltaofTemperature"
UNIT["degK"] = 1.
UNIT["degC"] = 1.
UNIT["degF"] = 0.5555556
UNIT["degR"] = 0.5555556

# dim = "Temparaturevariationrate"
UNIT["degK/s"] = 1.
UNIT["degF/s"] = 0.5555556

# dim = "Temparaturegradiant"
UNIT["degK/m"] = 1.
UNIT["degK/km"] = 0.001
UNIT["degF/m"] = 0.5555556
UNIT["degF/km"] = 5.555556e-4

# dim = "DeciBell"
UNIT["dB"] = 1.

# dim = "EffectivePerceivedDeciBell"
UNIT["EPNdB"] = 1.

# dim = "Mass"
UNIT["kg"] = 1.
UNIT["g"] = 0.001
UNIT["lb"] = 0.4535924
UNIT["lbm"] = 0.4535924
UNIT["t"] = 1000.
UNIT["T"] = 1000.
UNIT["metricT"] = 1000.

# dim = "MassIndex"
UNIT["g/kg"] = 1.

# dim = "MasstoForceratio"
UNIT["kg/N"] = 1.
UNIT["g/N"] = 0.001
UNIT["g/kN"] = 0.000001
UNIT["kg/daN"] = 0.1
UNIT["kg/kN"] = 0.001

# dim = "SeatKilometer"
UNIT["seat.m"] = 1.
UNIT["seat.km"] = 1000.

# dim = "MassperSeat"
UNIT["kg/seat"] = 1.
UNIT["g/seat"] = 0.001

# dim = "MassperSeatandperDistance"
UNIT["kg/m/seat"] = 1.
UNIT["kg/NM/seat"] = 0.000540

# dim = "SpecificConsumptionvsThrust"
UNIT["kg/N/s"] = 1.
UNIT["kg/s/N"] = 1.
UNIT["kg/daN/h"] = 2.77778e-05
UNIT["kg/h/daN"] = 2.77778e-05
UNIT["lb/lbf/h"] = 0.000028327
UNIT["lb/h/lbf"] = 0.000028327

# dim = "CruiseSpecificPowerAirRange"
UNIT["m/W"] = 1.
UNIT["NM/kW"] = 1.852

# dim = "SpecificEnergyConsumption"
UNIT["J/N/s"] = 1.
UNIT["kJ/daN/h"] = 1.e3
UNIT["MJ/lbf/h"] = 1.e6
UNIT["kW/daN"] = 100

# dim = "SpecificConsumptionvsPower"
UNIT["kg/W/s"] = 1.
UNIT["g/W/s"] = 1.e-3
UNIT["g/W/h"] = 1.e-3/3600
UNIT["g/kW/h"] = 1.e-6/3600
UNIT["kg/kW/h"] = 2.77778e-07
UNIT["lb/shp/h"] = 1.68969e-07

# dim = "Force"
UNIT["N"] = 1.
UNIT["kN"] = 1000.
UNIT["lbf"] = 4.4482198
UNIT["klbf"] = 4448.2198
UNIT["daN"] = 10.
UNIT["kgf"] = 9.8066502

# dim = "Pressure"
UNIT["Pa"] = 1.
UNIT["kPa"] = 1000.
UNIT["MPa"] = 1000000.
UNIT["kgf/m2"] = 9.8066502
UNIT["atm"] = 101325.
UNIT["bar"] = 100000.
UNIT["mbar"] = 100.
UNIT["psi"] = 6895.
UNIT["N/m2"] = 1.
UNIT["daN/m2"] = 10.

# dim = "Pressurevariationrate"
UNIT["Pa/s"] = 1.
UNIT["atm/s"] = 101325.
UNIT["bar/s"] = 100000.

# dim = "VolumetricMass"
UNIT["kg/m3"] = 1.
UNIT["kg/l"] = 1000.
UNIT["lb/ft3"] = 16.018499

# dim = "MassSensitivity"
UNIT["1/kg"] = 1.
UNIT["%/kg"] = 0.01
UNIT["%/ton"] = 0.01 * 0.001

# dim = "VolumetricMassFlow"
UNIT["kg/m3/s"] = 1.
UNIT["lb/ft3/s"] = 16.018499

# dim = "StandardUnit"
UNIT["si"] = 1.
UNIT["std"] = 1.
UNIT["uc"] = 1.
UNIT["cu"] = 1.

# dim = "Angle"
UNIT["rad"] = 1.
UNIT["deg"] = 0.0174533

# dim = "Volume"
UNIT["m3"] = 1.
UNIT["dm3"] = 0.001
UNIT["cm3"] = 0.000001
UNIT["litres"] = 0.001
UNIT["litre"] = 0.001
UNIT["l"] = 0.001
UNIT["L"] = 0.001
UNIT["ft3"] = 0.0283168

# dim = "VolumeFlow"
UNIT["m3/s"] = 1.
UNIT["litre/s"] = 0.001
UNIT["l/s"] = 0.001
UNIT["L/s"] = 0.001
UNIT["ft3/s"] = 0.0283168
UNIT["m3/min"] = 60.
UNIT["litre/min"] = 0.06
UNIT["l/min"] = 0.06
UNIT["L/min"] = 0.06
UNIT["ft3/min"] = 1.699008
UNIT["m3/h"] = 3600.
UNIT["litre/h"] = 3.6
UNIT["l/h"] = 3.6
UNIT["L/h"] = 3.6
UNIT["ft3/h"] = 101.94048

# dim = "VolumeCoefficient"
UNIT["m2/kN"] = 1.

# dim = "MachNumber"
UNIT["Mach"] = 1.
UNIT["mach"] = 1.

# dim = "DragCount"
UNIT["cx"] = 1.
UNIT["dc"] = 0.0001

# dim = "DragSensitivity"
UNIT["1/cx"] = 1.
UNIT["%/cx"] = 0.01
UNIT["%/dc"] = 0.01 * 10000.

# dim = "MachNumbervariationrate"
UNIT["Mach/s"] = 1.
UNIT["mach/s"] = 1.

# dim = "MassFlow"
UNIT["kg/s"] = 1.
UNIT["kg/h"] = 0.0002778
UNIT["lb/s"] = 0.4535924
UNIT["kg/min"] = 0.0166667
UNIT["lb/min"] = 0.00756
UNIT["lb/h"] = 0.000126

# dim = "Power"
UNIT["Watt"] = 1.
UNIT["W"] = 1.
UNIT["kW"] = 1.e3
UNIT["MW"] = 1.e6
UNIT["GW"] = 1.e9
UNIT["TW"] = 1.e12
UNIT["shp"] = 745.70001

# dim = "PowerDensity"
UNIT["W/kg"] = 1.
UNIT["kW/kg"] = 1.e3

# dim = "PowerDensityPerTime"
UNIT["kW/daN/h"] = 1 / 36.

# dim = "Euro"
UNIT["E"] = 1.
UNIT["Ec."] = 0.01
UNIT["kE"] = 1000.
UNIT["ME"] = 1000000.

# dim = "Cost"
UNIT["$"] = 1.
UNIT["$c."] = 0.01
UNIT["k$"] = 1000.
UNIT["M$"] = 1000000.

# dim = "HourlyCost"
UNIT["$/h"] = 1.

# dim = "Utilisation"
UNIT["trip/year"] = 1.

# dim = "TripCost"
UNIT["$/vol"] = 1.
UNIT["$/trip"] = 1.

# dim = "CosttoDistanceratio"
UNIT["$/km"] = 1.
UNIT["$/NM"] = 0.5399568

# dim = "CostDistancePax"
UNIT["$/km/pax"] = 1.
UNIT["$/NM/pax"] = 0.5399568

# dim = "Lineic"
UNIT["1/m"] = 1.

# dim = "Viscosity"
UNIT["Poises"] = 1.
UNIT["Pl.10e6"] = 0.000001

# dim = "SpecificCost"
UNIT["$/pax/km"] = 1.
UNIT["$/pax/NM"] = 0.5399568

# dim = "InverseAnglular"
UNIT["1/rad"] = 1.
UNIT["1/deg"] = 57.29578

# dim = "InversesquaredAngular"
UNIT["1/rad2"] = 1.

# dim = "MassicDistance"
UNIT["m/kg"] = 1.
UNIT["km/t"] = 1.
UNIT["km/kg"] = 1000.
UNIT["NM/t"] = 1.852
UNIT["NM/kg"] = 1852.
UNIT["NM/lb"] = 4082.8923

# dim = "EnergyDistance"
UNIT["J/m"] = 1.
UNIT["kWh/km"] = 3600

# dim = "DistanceEnergy"
UNIT["m/J"] = 1.
UNIT["km/kWh"] = 1/3600

# dim = "SurfacicMass"
UNIT["kg/m2"] = 1.
UNIT["lb/ft2"] = 4.8825102

# dim = "LineicMass"
UNIT["kg/m"] = 1.
UNIT["kg/km"] = 0.001
UNIT["lb/m"] = 0.4535924

# dim = "Momentum"
UNIT["N.m"] = 1.
UNIT["daN.m"] = 10.
UNIT["kgf.m"] = 9.8066502
UNIT["lbf.ft"] = 1.3558174

# dim = "InertiaMomentum"
UNIT["kg.m2"] = 1.
UNIT["lb.m2"] = 0.4535924

# dim = "AngularVelocity"
UNIT["rad/s"] = 1.
UNIT["deg/s"] = 0.0174533
UNIT["rpm"] = 0.1047198

# dim = "One_over_Second_square"
UNIT["1/s2"] = 1.

# dim = "AngularAcceleration"
UNIT["rad/s2"] = 1.
UNIT["deg/s2"] = 0.0174533
UNIT["rpm/s"] = 0.1047198

# dim = "Energy"
UNIT["J"] = 1.
UNIT["kJ"] = 1.e3
UNIT["MJ"] = 1.e6
UNIT["GJ"] = 1.e9
UNIT["TJ"] = 1.e12
UNIT["Wh"] = 3600.
UNIT["kWh"] = 3600.e3
UNIT["MWh"] = 3600.e6
UNIT["GWh"] = 3600.e9
UNIT["TWh"] = 3600.e12

# dim = "EnergyDensity"
UNIT["J/kg"] = 1.
UNIT["kJ/kg"] = 1.e3
UNIT["MJ/kg"] = 1.e6
UNIT["GJ/kg"] = 1.e9
UNIT["TJ/kg"] = 1.e12
UNIT["Wh/kg"] = 3600.
UNIT["kWh/kg"] = 3600.e3
UNIT["MWh/kg"] = 3600.e6
UNIT["GWh/kg"] = 3600.e9
UNIT["TWh/kg"] = 3600.e12

# dim = "MassicEnergy"
UNIT["J/kg"] = 1.
UNIT["kJ/kg"] = 1000.
UNIT["MJ/kg"] = 1000000.
UNIT["btu/lb"] = 2325.9612

# dim = "FuelCost"
UNIT["$/l"] = 1.
UNIT["$/gal"] = 0.264173
UNIT["$/USgal"] = 0.264173
UNIT["$/USbrl"] = 0.00838644

# dim = "BatteryMassCost"
UNIT["$/kg"] = 1.
UNIT["$/T"] = 1.e-3

# dim = "BatteryEnergyCost"
UNIT["$/kWh"] = 1. / UNIT['kWh']

# dim = "nodimension"
UNIT["sd"] = 1
UNIT["adim"] = 1
UNIT["no_dim"] = 1
UNIT["nd"] = 1
UNIT["%"] = 0.01
UNIT["%/%"] = 1.

# dim = "integer"
UNIT["integer"] = 1
UNIT["int"] = 1
UNIT["entier"] = 1
UNIT["numeric"] = 1

# dim = "variouscounts"
UNIT["aircraft"] = 1
UNIT["engine"] = 1
UNIT["pilot"] = 1
UNIT["attendant"] = 1
UNIT["trolley"] = 1
UNIT["toilet"] = 1
UNIT["seat"] = 1
UNIT["door"] = 1
UNIT["wheel"] = 1

# dim = "string"
UNIT["string"] = 1
UNIT["text"] = 1

# dim = "textdate"
UNIT["text_date"] = 1

# dim = "GlobalWarmingPotential_km"
UNIT["W/m2/km/year"] = 1.
UNIT["mW/m2/km/year"] = 1.e-3
UNIT["1e-3.W/m2/km/year"] = 1.e-3
UNIT["uW/m2/km/year"] = 1.e-6
UNIT["1e-6.W/m2/km/year"] = 1.e-6
UNIT["1e-12.W/m2/km/year"] = 1.e-12

# dim = "GlobalWarmingPotential_kg"
UNIT["W/m2/kg/year"] = 1.
UNIT["mW/m2/kg/year"] = 1.e-3
UNIT["1e-3.W/m2/kg/year"] = 1.e-3
UNIT["uW/m2/kg/year"] = 1.e-6
UNIT["1e-6.W/m2/kg/year"] = 1.e-6
UNIT["1e-12.W/m2/kg/year"] = 1.e-12

# dim = "GlobalWarmingPotential"
UNIT["W/m2/year"] = 1.
UNIT["mW/m2/year"] = 1.e-3
UNIT["1e-3.W/m2/year"] = 1.e-3
UNIT["uW/m2/year"] = 1.e-6
UNIT["1e-6.W/m2/year"] = 1.e-6
UNIT["1e-12.W/m2/year"] = 1.e-12

# dim = "GlobalWarmingPotential"
UNIT["W/m2"] = 1.
UNIT["mW/m2"] = 1.e-3
UNIT["1e-3.W/m2"] = 1.e-3
UNIT["uW/m2"] = 1.e-6
UNIT["1e-6.W/m2"] = 1.e-6
UNIT["1e-12.W/m2"] = 1.e-12

# dim = "SpecificMassicEmission"
UNIT["Pa.m3/kg"] = 1.
UNIT["bar.l/kg"] = 100.

# dim = "SpecificMassicEmission"
UNIT["g/seat/m"] = 1.
UNIT["g/seat/km"] = 0.001

# dim = "SpecificVolumicConsumption"
UNIT["m3/seat/m"] = 1.
UNIT["l/seat/100km"] = 0.01

# dim = "CO2metric"
UNIT["kg/m/m^0.48"] = 1.
UNIT["kg/m/m0.48"] = 1.
UNIT["kg/km/m^0.48"] = 0.001
UNIT["kg/km/m0.48"] = 0.001
UNIT["kg/NM/m^0.48"] = 1./1852.
UNIT["kg/NM/m0.48"] = 1./1852.

# dim = "GlobalWarmingTemperature"
UNIT["K/m2/km/year"] = 1.
UNIT["1e-6.K/m2/km/year"] = 1.e-6
UNIT["1e-12.K/m2/km/year"] = 1.e-12

# dim = "DataStructure"
UNIT["structure"] = 1
UNIT["dict"] = 1
UNIT["array"] = 1



# Conversion functions
#-------------------------------------------------------------------------
def convert_from(ulab, val):
    """Convert val expressed in ulab to corresponding standard unit
    :param ulab: unit label. ex: 'NM' for Nautical miles
    :param val: the value to convert to SI units
    :return: val converted to SI units
    """

    if isinstance(val, (type(None), str)):
        return val
    if isinstance(val, list):
        return [convert_from(ulab, v) for v in val]
    if isinstance(val, tuple):
        return (convert_from(ulab, v) for v in val)
    if isinstance(val, numpy.ndarray):
        return numpy.array([convert_from(ulab, v) for v in val])
    if isinstance(val, dict):
        dic_val = deepcopy(val)
        for k, v in dic_val.items():
            dic_val[k] = convert_from(ulab, v)
        return dic_val
    return val * UNIT[ulab]


def convert_to(ulab, val):
    # Convert val expressed in standard unit to ulab
    if isinstance(val, (type(None), str)):
        return val
    if isinstance(val, list):
        return [convert_to(ulab, v) for v in val]
    if isinstance(val, tuple):
        return tuple([convert_to(ulab, v) for v in val])
    if isinstance(val, numpy.ndarray):
        return numpy.array([convert_to(ulab, v) for v in val])
    if isinstance(val, dict):
        dic_val = deepcopy(val)
        for k, v in dic_val.items():
            dic_val[k] = convert_to(ulab, v)
        return dic_val
    if ulab in ["integer", "int", "entier", "numeric"]:
        val = int(val)
    return val / UNIT[ulab]


def pretty_print(fmt, v, u):
    w = str(convert_to(u, v))
    s = eval("'"+fmt+"'"+"%"+w)
    return s+" "+u
