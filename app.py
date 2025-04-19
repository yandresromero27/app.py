import os
import sys
import json
import time
import random
import asyncio
import pyfiglet
import numpy as np
from pathlib import Path
from quotexapi.expiration import (
    timestamp_to_date,
    get_timestamp_days_ago
)
from quotexapi.utils.processor import (
    process_candles,
    get_color,
    aggregate_candle
)
from quotexapi.config import credentials
from quotexapi.stable_api import Quotex

__author__ = "Cleiton Leonel Creton"
__version__ = "1.1.0"  # Version actualizada

__message__ = f"""
Use in moderation, because management is everything!
Support: cleiton.leonel@gmail.com or +55 (27) 9 9577-2291
"""

USER_AGENT = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0"

custom_font = pyfiglet.Figlet(font="ansi_shadow")
ascii_art = custom_font.renderText("PyQuotex")
art_effect = f"""{ascii_art}

        author: {__author__} versão: {__version__}
        {__message__}
"""

print(art_effect)

# After the first access, the user's credentials will be available in this function.
email, password = credentials()

client = Quotex(
    email=email,
    password=password,
    lang="pt",  # Default pt -> Português.
)

# Smart Money Concepts & Advanced Pattern Detection Functions

def detect_liquidity_grab(candles, lookback=5):
    """Detect liquidity grab patterns (stop hunts)"""
    if len(candles) < lookback + 3:
        return None
    
    recent_candles = candles[-lookback-3:]
    
    # Check for bullish liquidity grab (price dips below recent lows then reverses up)
    recent_lows = [candle['low'] for candle in recent_candles[:-3]]
    min_low = min(recent_lows)
    
    last_three = recent_candles[-3:]
    
    # MEJORA: Confirmación adicional de volumen para liquidity grab
    volume_increase = False
    if 'volume' in candles[0]:
        avg_volume = np.mean([c.get('volume', 0) for c in recent_candles[:-3]])
        if last_three[0].get('volume', 0) > avg_volume * 1.5:
            volume_increase = True
    
    # Check if price went below recent lows (grabbed liquidity) then reversed
    if (last_three[0]['low'] < min_low and 
        last_three[1]['close'] > last_three[1]['open'] and 
        last_three[2]['close'] > last_three[2]['open']):
        if not volume_increase:
            return "call", 0.8  # Bullish after liquidity grab, medium confidence
        return "call", 1.0  # Higher confidence with volume
    
    # Check for bearish liquidity grab (price spikes above recent highs then reverses down)
    recent_highs = [candle['high'] for candle in recent_candles[:-3]]
    max_high = max(recent_highs)
    
    if (last_three[0]['high'] > max_high and 
        last_three[1]['close'] < last_three[1]['open'] and 
        last_three[2]['close'] < last_three[2]['open']):
        if not volume_increase:
            return "put", 0.8  # Bearish after liquidity grab, medium confidence
        return "put", 1.0  # Higher confidence with volume
    
    return None, 0

def detect_order_block(candles, lookback=10):
    """Identify order blocks - areas where smart money initiates positions"""
    if len(candles) < lookback + 3:
        return None, 0
    
    recent_candles = candles[-lookback:]
    
    # Find strong momentum candles that might indicate smart money entering
    for i in range(len(recent_candles) - 3):
        # Look for a strong momentum candle
        candle = recent_candles[i]
        body_size = abs(candle['close'] - candle['open'])
        
        # Check if it's a strong candle
        avg_body = np.mean([abs(c['close'] - c['open']) for c in recent_candles])
        
        # MEJORA: Más refinamiento para identificar order blocks verdaderos
        if body_size > 1.7 * avg_body:  # Aumentado a 1.7x para mayor precisión
            # Bullish order block - look for price returning to the base
            if candle['close'] > candle['open']:  # Bullish candle
                # Modified zone calculation for more precise order blocks
                order_block_low = candle['open']
                order_block_high = candle['open'] + (body_size * 0.4)  # Narrowed zone
                
                # Check if recent price has returned to the order block zone
                latest_low = recent_candles[-1]['low']
                latest_high = recent_candles[-1]['high']
                
                # MEJORA: Verificar si el precio ha tocado pero no atravesado la zona
                if latest_low <= order_block_high and latest_high >= order_block_low:
                    # Check if we've had a reaction at this level
                    if recent_candles[-1]['close'] > recent_candles[-1]['open']:
                        return "call", 0.9  # Bullish opportunity at order block with reaction
            
            # Bearish order block
            elif candle['close'] < candle['open']:  # Bearish candle
                # Modified zone calculation
                order_block_low = candle['open'] - (body_size * 0.4)  # Narrowed zone
                order_block_high = candle['open']
                
                # Check if recent price has returned to the order block zone
                latest_low = recent_candles[-1]['low']
                latest_high = recent_candles[-1]['high']
                
                if latest_low <= order_block_high and latest_high >= order_block_low:
                    # Check if we've had a reaction at this level
                    if recent_candles[-1]['close'] < recent_candles[-1]['open']:
                        return "put", 0.9  # Bearish opportunity at order block with reaction
    
    return None, 0

def detect_fair_value_gap(candles):
    """Detect Fair Value Gaps (FVG) - gaps in price that are likely to be filled"""
    if len(candles) < 4:
        return None, 0
    
    # Look at recent candles
    c1 = candles[-4]  # Third last candle
    c2 = candles[-3]  # Second last candle
    c3 = candles[-2]  # Last complete candle
    
    # MEJORA: Calcular tamaño del gap para medir importancia
    gap_size_bullish = c3['low'] - c1['high'] if c1['high'] < c3['low'] else 0
    gap_size_bearish = c1['low'] - c3['high'] if c1['low'] > c3['high'] else 0
    
    # Normalizar por ATR para medir significancia relativa
    price_range = np.mean([c['high'] - c['low'] for c in candles[-10:]])
    
    # Bullish FVG - gap between the high of first candle and low of third candle
    if gap_size_bullish > 0:
        # MEJORA: Solo considerar gaps significativos
        if gap_size_bullish > price_range * 0.3:
            # Price is likely to come down to fill this gap
            confidence = min(0.7 + (gap_size_bullish / price_range) * 0.3, 0.95)
            return "put", confidence
    
    # Bearish FVG - gap between the low of first candle and high of third candle
    if gap_size_bearish > 0:
        # MEJORA: Solo considerar gaps significativos
        if gap_size_bearish > price_range * 0.3:
            # Price is likely to go up to fill this gap
            confidence = min(0.7 + (gap_size_bearish / price_range) * 0.3, 0.95)
            return "call", confidence
    
    return None, 0

def detect_breaker_block(candles, lookback=15):
    """Detect breaker blocks - former support/resistance areas that have been broken and retested"""
    if len(candles) < lookback:
        return None, 0
    
    # Get high and low values
    highs = [candle['high'] for candle in candles]
    lows = [candle['low'] for candle in candles]
    
    # Find swing points in the first part of the candles
    first_part = candles[:lookback//2]
    
    # Find potential support/resistance levels
    resistance_levels = []
    support_levels = []
    
    # MEJORA: Método mejorado para encontar swing points
    for i in range(2, len(first_part)-2):
        # Swing high - más restrictivo
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
            highs[i] > highs[i+1] and highs[i] > highs[i+2] and
            highs[i] - np.mean([highs[i-2], highs[i-1], highs[i+1], highs[i+2]]) > 
            np.std([candle['high'] for candle in first_part]) * 0.5):
            resistance_levels.append(highs[i])
        
        # Swing low - más restrictivo
        if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
            lows[i] < lows[i+1] and lows[i] < lows[i+2] and
            np.mean([lows[i-2], lows[i-1], lows[i+1], lows[i+2]]) - lows[i] > 
            np.std([candle['low'] for candle in first_part]) * 0.5):
            support_levels.append(lows[i])
    
    # If we found levels, check if they've been broken and retested
    if resistance_levels:
        # Check if resistance was broken (price closed above it) and now retesting from above
        latest_close = candles[-1]['close']
        latest_low = candles[-1]['low']
        
        for level in resistance_levels:
            # Find if there was a break of this level
            for i in range(lookback//2, len(candles)-3):
                if candles[i]['close'] > level and candles[i+1]['close'] > level:
                    # Level was broken, now check if current price is retesting it
                    # MEJORA: Rango más preciso para retest
                    if latest_low <= level * 1.003 and latest_close > level:  # Within 0.3% of level
                        # Check for rejection (wick below but close above)
                        if latest_low < level and latest_close > level:
                            return "call", 0.9  # High confidence on clear rejection
                        return "call", 0.8  # Buy on retest of broken resistance
    
    if support_levels:
        # Check if support was broken (price closed below it) and now retesting from below
        latest_close = candles[-1]['close']
        latest_high = candles[-1]['high']
        
        for level in support_levels:
            # Find if there was a break of this level
            for i in range(lookback//2, len(candles)-3):
                if candles[i]['close'] < level and candles[i+1]['close'] < level:
                    # Level was broken, now check if current price is retesting it
                    # MEJORA: Rango más preciso para retest
                    if latest_high >= level * 0.997 and latest_close < level:  # Within 0.3% of level
                        # Check for rejection (wick above but close below)
                        if latest_high > level and latest_close < level:
                            return "put", 0.9  # High confidence on clear rejection
                        return "put", 0.8  # Sell on retest of broken support
    
    return None, 0

def calculate_rsi(candles, periods=14):
    """Calculate RSI (Relative Strength Index)"""
    if len(candles) < periods + 1:
        return None
    
    # Extract closing prices
    closes = [candle['close'] for candle in candles]
    
    # Calculate price changes
    deltas = np.diff(closes)
    
    # Calculate gains and losses
    gains = np.copy(deltas)
    losses = np.copy(deltas)
    
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    # Calculate average gains and losses
    avg_gain = np.mean(gains[:periods])
    avg_loss = np.mean(losses[:periods])
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_adaptive_rsi(candles, base_period=14):
    """Calculate RSI with adaptive period based on volatility"""
    # MEJORA: RSI adaptativo basado en volatilidad del mercado
    if len(candles) < base_period + 20:
        return calculate_rsi(candles, base_period)
    
    # Calculate recent volatility
    closes = [candle['close'] for candle in candles[-30:]]
    volatility = np.std(closes) / np.mean(closes)
    
    # Adjust RSI period based on volatility
    if volatility > 0.015:  # High volatility
        adjusted_period = max(8, int(base_period * 0.7))  # Shorter period for fast response
    elif volatility < 0.005:  # Low volatility
        adjusted_period = min(21, int(base_period * 1.3))  # Longer period to reduce noise
    else:
        adjusted_period = base_period
    
    return calculate_rsi(candles, adjusted_period)

def detect_divergence(candles, lookback=20):
    """Detect RSI divergences - powerful reversal signals"""
    if len(candles) < lookback + 14:  # Need extra candles for RSI calculation
        return None, 0
    
    # Get closing prices and calculate RSIs
    section = candles[-lookback-14:]
    rsi_values = []
    
    # MEJORA: Usar RSI adaptativo
    for i in range(lookback):
        window = section[i:i+14+1]
        rsi = calculate_adaptive_rsi(window)
        if rsi is not None:
            rsi_values.append(rsi)
    
    if len(rsi_values) < 10:  # Need enough RSI values
        return None, 0
    
    price_highs = []
    price_lows = []
    rsi_highs = []
    rsi_lows = []
    
    # Find swing highs and lows in price and RSI
    for i in range(2, len(rsi_values)-2):
        # Price swing high
        if candles[-lookback+i]['high'] > candles[-lookback+i-1]['high'] and candles[-lookback+i]['high'] > candles[-lookback+i-2]['high'] and \
           candles[-lookback+i]['high'] > candles[-lookback+i+1]['high'] and candles[-lookback+i]['high'] > candles[-lookback+i+2]['high']:
            price_highs.append((i, candles[-lookback+i]['high']))
        
        # Price swing low
        if candles[-lookback+i]['low'] < candles[-lookback+i-1]['low'] and candles[-lookback+i]['low'] < candles[-lookback+i-2]['low'] and \
           candles[-lookback+i]['low'] < candles[-lookback+i+1]['low'] and candles[-lookback+i]['low'] < candles[-lookback+i+2]['low']:
            price_lows.append((i, candles[-lookback+i]['low']))
        
        # RSI swing high
        if rsi_values[i] > rsi_values[i-1] and rsi_values[i] > rsi_values[i-2] and \
           rsi_values[i] > rsi_values[i+1] and rsi_values[i] > rsi_values[i+2]:
            rsi_highs.append((i, rsi_values[i]))
        
        # RSI swing low
        if rsi_values[i] < rsi_values[i-1] and rsi_values[i] < rsi_values[i-2] and \
           rsi_values[i] < rsi_values[i+1] and rsi_values[i] < rsi_values[i+2]:
            rsi_lows.append((i, rsi_values[i]))
    
    # MEJORA: Comprobar la distancia entre los puntos de divergencia para mayor validez
    def check_points_proximity(points1, points2):
        if len(points1) < 2 or len(points2) < 2:
            return False
        # Check if the latest points are close in time
        return abs(points1[-1][0] - points2[-1][0]) <= 3 and abs(points1[-2][0] - points2[-2][0]) <= 3
    
    # Check for bearish divergence (price higher, RSI lower)
    if len(price_highs) >= 2 and len(rsi_highs) >= 2:
        price_higher = price_highs[-1][1] > price_highs[-2][1]
        rsi_lower = rsi_highs[-1][1] < rsi_highs[-2][1]
        
        if price_higher and rsi_lower and check_points_proximity(price_highs, rsi_highs):
            # MEJORA: Añadir filtro de nivel RSI para evitar falsas señales
            if rsi_highs[-1][1] > 65:  # Solo divergencias en RSI alto
                confidence = 0.7
                # Stronger signal if RSI is overbought
                if rsi_highs[-1][1] > 75:
                    confidence = 0.9
                return "put", confidence  # Bearish divergence
    
    # Check for bullish divergence (price lower, RSI higher)
    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        price_lower = price_lows[-1][1] < price_lows[-2][1]
        rsi_higher = rsi_lows[-1][1] > rsi_lows[-2][1]
        
        if price_lower and rsi_higher and check_points_proximity(price_lows, rsi_lows):
            # MEJORA: Añadir filtro de nivel RSI para evitar falsas señales
            if rsi_lows[-1][1] < 35:  # Solo divergencias en RSI bajo
                confidence = 0.7
                # Stronger signal if RSI is oversold
                if rsi_lows[-1][1] < 25:
                    confidence = 0.9
                return "call", confidence  # Bullish divergence
    
    return None, 0

def identify_market_structure(candles, lookback=20):
    """Identify market structure shifts"""
    if len(candles) < lookback:
        return None, 0
    
    # Extract highs and lows
    highs = [candle['high'] for candle in candles[-lookback:]]
    lows = [candle['low'] for candle in candles[-lookback:]]
    
    # MEJORA: Cálculo de tendencia por media móvil
    ma_short = np.mean([candle['close'] for candle in candles[-8:]])
    ma_medium = np.mean([candle['close'] for candle in candles[-21:]])
    
    # Current close price
    current_close = candles[-1]['close']
    
    # Find swing highs and lows with improved algorithm
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(highs) - 2):
        # Swing high
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append((i, highs[i]))
        
        # Swing low
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append((i, lows[i]))
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        # MEJORA: Usar análisis de MA si no hay suficientes swing points
        if ma_short > ma_medium and current_close > ma_short:
            return "call", 0.6  # Uptrend based on MA
        elif ma_short < ma_medium and current_close < ma_short:
            return "put", 0.6  # Downtrend based on MA
        return None, 0
    
    # Check for Higher Highs and Higher Lows (Uptrend)
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        # MEJORA: Calcular fuerza de tendencia basada en el tamaño de los swings
        hhhl_strength = 0
        llhl_strength = 0
        
        if swing_highs[-1][1] > swing_highs[-2][1] and swing_lows[-1][1] > swing_lows[-2][1]:
            # Calculate uptrend strength
            high_change = (swing_highs[-1][1] / swing_highs[-2][1] - 1) * 100
            low_change = (swing_lows[-1][1] / swing_lows[-2][1] - 1) * 100
            hhhl_strength = (high_change + low_change) / 2
            
            confidence = min(0.6 + hhhl_strength / 10, 0.9)
            
            # Filter by MA confirmation
            if ma_short > ma_medium:
                return "call", confidence  # Uptrend structure
        
        # Check for Lower Highs and Lower Lows (Downtrend)
        if swing_highs[-1][1] < swing_highs[-2][1] and swing_lows[-1][1] < swing_lows[-2][1]:
            # Calculate downtrend strength
            high_change = (1 - swing_highs[-1][1] / swing_highs[-2][1]) * 100
            low_change = (1 - swing_lows[-1][1] / swing_lows[-2][1]) * 100
            llhl_strength = (high_change + low_change) / 2
            
            confidence = min(0.6 + llhl_strength / 10, 0.9)
            
            # Filter by MA confirmation
            if ma_short < ma_medium:
                return "put", confidence  # Downtrend structure
        
        # MEJORA: Detección mejorada de cambios estructurales
        # Check for potential structure break - more strict criteria
        if (swing_highs[-1][1] < swing_highs[-2][1] and 
            swing_lows[-1][1] > swing_lows[-2][1] and
            current_close > swing_highs[-2][1]):
            # Confirmed break of structure from down to up
            return "call", 0.85
        
        if (swing_highs[-1][1] > swing_highs[-2][1] and 
            swing_lows[-1][1] < swing_lows[-2][1] and
            current_close < swing_lows[-2][1]):
            # Confirmed break of structure from up to down
            return "put", 0.85
    
    return None, 0

# NUEVA FUNCIÓN: Análisis de volatilidad
def analyze_volatility(candles, lookback=20):
    """Analyze current market volatility to avoid ranging markets"""
    if len(candles) < lookback:
        return None
    
    # Calculate average true range (ATR)
    tr_values = []
    for i in range(1, lookback):
        high = candles[-i]['high']
        low = candles[-i]['low']
        prev_close = candles[-(i+1)]['close']
        
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)
    
    atr = np.mean(tr_values)
    
    # Calculate average price
    avg_price = np.mean([candle['close'] for candle in candles[-lookback:]])
    
    # Relative ATR as percentage of price
    rel_atr = atr / avg_price * 100
    
    if rel_atr < 0.1:
        return "low"  # Very low volatility, avoid trading
    elif rel_atr < 0.3:
        return "moderate"  # Cautious trading
    else:
        return "high"  # Good volatility for trading
        
# NUEVA FUNCIÓN: Detección de zonas de acumulación/distribución
def detect_accumulation_distribution(candles, lookback=30):
    """Detect smart money accumulation or distribution phases"""
    if len(candles) < lookback:
        return None, 0
    
    # Calculate money flow for each candle
    money_flow = []
    for i in range(lookback):
        candle = candles[-lookback+i]
        # Money Flow Multiplier: ((Close - Low) - (High - Close)) / (High - Low)
        if candle['high'] == candle['low']:  # Avoid division by zero
            mf_mult = 0
        else:
            mf_mult = ((candle['close'] - candle['low']) - (candle['high'] - candle['close'])) / (candle['high'] - candle['low'])
        
        # Assume volume is 1 if not available
        volume = candle.get('volume', 1)
        money_flow.append(mf_mult * volume)
    
    # Calculate 14-period sum
    if len(money_flow) < 14:
        return None, 0
    
    adl_periods = [sum(money_flow[i:i+14]) for i in range(len(money_flow)-14+1)]
    
    # Look for accumulation (positive flow) or distribution (negative flow)
    if len(adl_periods) < 2:
        return None, 0
    
    recent_flow = adl_periods[-1]
    # Strong positive flow indicates accumulation
    if recent_flow > 0 and recent_flow > 1.5 * np.mean(adl_periods):
        return "call", 0.7
    # Strong negative flow indicates distribution
    elif recent_flow < 0 and recent_flow < 1.5 * np.mean([abs(p) for p in adl_periods if p < 0]):
        return "put", 0.7
    
    return None, 0

def smart_money_analysis(candles):
    """Comprehensive Smart Money analysis combining multiple concepts with weighted signals"""
    if len(candles) < 30:  # Need enough data for analysis
        return None, "Not enough data for analysis", 0
    
    # MEJORA: Sistema de pesos y confianza para señales
    signals = []
    
    # Check volatility first - exit if too low
    volatility = analyze_volatility(candles)
    if volatility == "low":
        return None, "Market volatility too low for quality signals", 0
    
    # Apply different smart money concepts with confidence levels
    liq_signal, liq_conf = detect_liquidity_grab(candles)
    if liq_signal:
        signals.append((liq_signal, liq_conf * 1.2))  # High weight for liquidity grabs
    
    ob_signal, ob_conf = detect_order_block(candles)
    if ob_signal:
        signals.append((ob_signal, ob_conf * 1.1))  # High weight for order blocks
    
    fvg_signal, fvg_conf = detect_fair_value_gap(candles)
    if fvg_signal:
        signals.append((fvg_signal, fvg_conf * 0.9))  # Lower weight for FVGs
    
    breaker_signal, breaker_conf = detect_breaker_block(candles)
    if breaker_signal:
        signals.append((breaker_signal, breaker_conf * 1.15))  # Very high weight for breaker blocks
    
    div_signal, div_conf = detect_divergence(candles)
    if div_signal:
        signals.append((div_signal, div_conf * 1.05))  # High weight for divergences
    
    struct_signal, struct_conf = identify_market_structure(candles)
    if struct_signal:
        signals.append((struct_signal, struct_conf * 1.0))  # Medium weight for structure
    
    # New signal: accumulation/distribution
    accum_signal, accum_conf = detect_accumulation_distribution(candles)
    if accum_signal:
        signals.append((accum_signal, accum_conf * 0.95))  # Medium weight
    
    if not signals:
        return None, "No clear Smart Money signal", 0
    
    # MEJORA: Sistema de evaluación ponderada
    call_weight = sum([conf for signal, conf in signals if signal == "call"])
    put_weight = sum([conf for signal, conf in signals if signal == "put"])
    
    # Calculate strength based on number of signals and their confidence
    total_conf = max(call_weight, put_weight)
    signal_strength = min(total_conf / 1.5, 0.95)  # Cap at 0.95
    
    # Enhanced decision making
    if call_weight > put_weight * 1.5:  # Strong call bias
        return "call", f"Strong bullish Smart Money signals ({len([s for s,c in signals if s == 'call'])} signals)", signal_strength
    elif put_weight > call_weight * 1.5:  # Strong put bias
        return "put", f"Strong bearish Smart Money signals ({len([s for s,c in signals if s == 'put'])} signals)", signal_strength
    elif call_weight > put_weight:
        return "call", f"Moderate bullish bias ({len([s for s,c in signals if s == 'call'])} signals)", signal_strength * 0.9
    elif put_weight > call_weight:
        return "put", f"Moderate bearish bias ({len([s for s,c in signals if s == 'put'])} signals)", signal_strength * 0.9
    else:
        return None, "Mixed signals - no clear direction", 0


async def connect(attempts=5):
    check, reason = await client.connect()
    if not check:
        attempt = 0
        while attempt <= attempts:
            if not await client.check_connect():
                check, reason = await client.connect()
                if check:
                    print("Reconnected successfully!!!")
                    break
                else:
                    print("Error when reconnecting.")
                    attempt += 1
                    if Path(os.path.join(".", "session.json")).is_file():
                        Path(os.path.join(".", "session.json")).unlink()
                    print(f"Reconnecting, attempt {attempt} de {attempts}")
            elif not check:
                attempt += 1
            else:
                break

            await asyncio.sleep(5)

        return check, reason

    print(reason)

    return check, reason

# Risk Management System - MEJORADO
class RiskManager:
    def __init__(self, initial_balance):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_drawdown = 0
        self.win_streak = 0
        self.loss_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        # MEJORA: Variables adicionales para gestión avanzada de riesgo
        self.hourly_performance = []  # Track performance by hour
        self.asset_performance = {}   # Track performance by asset
        self.last_trade_time = None
        self.daily_loss_limit = initial_balance * 0.05  # 5% daily stop loss
        self.daily_loss = 0
        self.confidence_multiplier = 1.0
    
    def update(self, result, profit, asset=None, confidence=0):
        self.total_trades += 1
        self.current_balance += profit
        current_time = time.time()
        
        # Update daily loss tracker
        if profit < 0:
            self.daily_loss += abs(profit)
        
        # Store time performance data
        hour = time.localtime(current_time).tm_hour
        self.hourly_performance.append((hour, result, profit))
        
        # Store asset performance
        if asset:
            if asset not in self.asset_performance:
                self.asset_performance[asset] = {"wins": 0, "losses": 0, "profit": 0}
            
            if result == "win":
                self.asset_performance[asset]["wins"] += 1
            else:
                self.asset_performance[asset]["losses"] += 1
            
            self.asset_performance[asset]["profit"] += profit
        
        # Update drawdown
        drawdown = (self.initial_balance - self.current_balance) / self.initial_balance * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        # Update streaks
        if result == "win":
            self.winning_trades += 1
            self.win_streak += 1
            self.loss_streak = 0
            if self.win_streak > self.max_win_streak:
                self.max_win_streak = self.win_streak
            
            # MEJORA: Ajuste de confianza tras victorias consecutivas
            if self.win_streak >= 3:
                self.confidence_multiplier = min(1.2, self.confidence_multiplier + 0.05)
        else:
            self.win_streak = 0
            self.loss_streak += 1
            if self.loss_streak > self.max_loss_streak:
                self.max_loss_streak = self.loss_streak
            
            # MEJORA: Ajuste de confianza tras pérdidas consecutivas
            if self.loss_streak >= 2:
                self.confidence_multiplier = max(0.8, self.confidence_multiplier - 0.1)
        
        self.last_trade_time = current_time
    
    def get_trade_amount(self, asset=None, confidence=0.7):
        """Dynamic position sizing based on account balance, performance metrics, and signal confidence"""
        # MEJORA: Sistema de sizing adaptativo más inteligente
        
        # Base amount starts at 1% of current balance
        base_amount = self.current_balance * 0.01
        
        # Adjust for win rate if we have enough trades
        if self.total_trades > 5:
            win_rate = self.winning_trades / self.total_trades
            
            # Adapt to win rate
            if win_rate > 0.65:
                base_amount *= 1.3
            elif win_rate < 0.4:
                base_amount *= 0.7
                
        # Adjust for streaks
        if self.win_streak >= 3:
            streak_multiplier = min(1.0 + (self.win_streak * 0.1), 1.5)  # Cap at 50% increase
            base_amount *= streak_multiplier
        elif self.loss_streak >= 2:
            # Reduce size after consecutive losses, more aggressively for longer streaks
            reduction = max(0.5, 1.0 - (self.loss_streak * 0.15))  # Min 50% reduction
            base_amount *= reduction
        
        # Factor in asset performance if available
        if asset and asset in self.asset_performance and sum(self.asset_performance[asset].values()) > 2:
            asset_wins = self.asset_performance[asset]["wins"]
            asset_losses = self.asset_performance[asset]["losses"]
            asset_total = asset_wins + asset_losses
            
            if asset_total >= 3:  # Only consider if we have enough data
                asset_win_rate = asset_wins / asset_total
                if asset_win_rate > 0.6:
                    base_amount *= 1.2  # Increase size for assets that perform well
                elif asset_win_rate < 0.4:
                    base_amount *= 0.8  # Decrease size for underperforming assets
        
        # Factor in signal confidence - key improvement
        confidence_adj = max(0.6, min(1.4, confidence * 1.5))  # Scale between 0.6x and 1.4x
        base_amount *= confidence_adj
        
        # Apply confidence multiplier based on recent performance
        base_amount *= self.confidence_multiplier
        
        # Risk limiting safeguards
        if self.daily_loss > self.daily_loss_limit * 0.7:  # If approaching daily loss limit
            base_amount *= 0.6  # Significantly reduce position size
        
        # Ensure minimum and maximum amounts
        min_amount = 5  # Minimum $5
        max_amount = self.current_balance * 0.03  # Maximum 3% of balance per trade
        
        return max(min_amount, min(base_amount, max_amount))
    
    def should_stop_trading(self):
        """Determine if trading should be halted based on risk parameters"""
        # MEJORA: Reglas más refinadas para pausar el trading
        
        # Stop if drawdown exceeds 12% (reduced from 15%)
        if self.max_drawdown > 12:
            return True, "Maximum drawdown exceeded"
        
        # Stop if we lose 4 trades in a row (reduced from 5)
        if self.loss_streak >= 4:
            return True, "Too many consecutive losses"
        
        # Stop if we hit daily loss limit
        if self.daily_loss >= self.daily_loss_limit:
            return True, "Daily loss limit reached"
        
        # Check for adverse hour performance
        if len(self.hourly_performance) >= 5:
            current_hour = time.localtime().tm_hour
            hour_trades = [t for t in self.hourly_performance if t[0] == current_hour]
            
            if len(hour_trades) >= 3:
                hour_results = [t[1] for t in hour_trades]
                if hour_results.count("win") / len(hour_results) < 0.3:
                    return True, f"Poor performance during current hour ({current_hour}:00)"
        
        return False, None
    
    def best_trading_assets(self):
        """Identify the best performing assets"""
        if not self.asset_performance:
            return None
            
        ranked_assets = []
        for asset, data in self.asset_performance.items():
            if data["wins"] + data["losses"] >= 3:  # Only consider with enough data
                win_rate = data["wins"] / (data["wins"] + data["losses"]) if (data["wins"] + data["losses"]) > 0 else 0
                profit_per_trade = data["profit"] / (data["wins"] + data["losses"]) if (data["wins"] + data["losses"]) > 0 else 0
                score = (win_rate * 0.7) + (profit_per_trade * 0.3 / 10)  # Weighted score
                ranked_assets.append((asset, score))
        
        if not ranked_assets:
            return None
            
        # Return top performers
        return [a[0] for a in sorted(ranked_assets, key=lambda x: x[1], reverse=True)]

# NUEVA FUNCIÓN: Filtro de horario para trading
def is_good_trading_time():
    """Check if current time is good for trading based on market activity"""
    now = time.localtime()
    hour = now.tm_hour
    weekday = now.tm_wday  # 0-6, Monday is 0
    
    # Avoid weekends
    if weekday >= 5:  # Saturday and Sunday
        return False
    
    # Best forex hours (simplified)
    if 2 <= hour <= 4:  # Asian session peak
        return True
    if 8 <= hour <= 11:  # London session peak
        return True
    if 14 <= hour <= 16:  # NY session peak
        return True
    if 19 <= hour <= 20:  # Sydney open
        return True
    
    # Other hours - moderate activity
    if 5 <= hour <= 7 or 12 <= hour <= 13 or 17 <= hour <= 18:
        return True
    
    # Avoid low liquidity hours
    return False

# ADVANCED AUTOMATED TRADING STRATEGY - MEJORADO
async def automated_trading():
    print("\n=== STARTING ENHANCED SMART MONEY AUTOMATED TRADING ===\n")
    
    check_connect, message = await client.connect()
    if not check_connect:
        print("Connection failed. Exiting...")
        return
    
    # Change to practice account for safety
    client.change_account("PRACTICE")
    initial_balance = await client.get_balance()
    print(f"Initial Balance: {initial_balance}")
    
    # Initialize risk manager
    risk_manager = RiskManager(initial_balance)
    
    # Trading parameters
    max_trades = 15   # Maximum number of trades to execute
    # MEJORA: Lista ampliada de activos con prioridad para pares más estables
    assets_to_trade = [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",  # Major forex pairs (priority)
        "USDCAD", "NZDUSD", "EURGBP", "EURJPY",
        "GBPJPY", "AUDJPY", "CADCHF", "EURCHF",
        "AUDCAD", "GBPCAD", "GBPAUD", "EURAUD"
    ]
    
    # MEJORA: Duración de operación adaptativa
    base_duration = 300    # Base trade duration in seconds (5 minutes)
    
    win_count = 0
    loss_count = 0
    
    try:
        for trade_num in range(max_trades):
            print(f"\n--- Trade {trade_num + 1}/{max_trades} ---")
            
            # MEJORA: Verificar si es buen momento para operar
            if not is_good_trading_time():
                print("Current time is not optimal for trading. Waiting for better conditions...")
                delay = random.randint(300, 600)  # Wait 5-10 minutes
                print(f"Waiting {delay} seconds")
                await asyncio.sleep(delay)
                continue
            
            # Check if we should stop trading based on risk management
            should_stop, reason = risk_manager.should_stop_trading()
            if should_stop:
                print(f"Risk management alert: {reason}")
                print("Trading stopped for safety.")
                break
            
            # MEJORA: Priorizar los activos con mejor rendimiento
            prioritized_assets = risk_manager.best_trading_assets()
            if prioritized_assets and len(prioritized_assets) >= 3:
                print("Using performance-based asset priority")
                # Use top performers but keep some randomness
                assets_to_try = prioritized_assets[:5] + random.sample(assets_to_trade, min(5, len(assets_to_trade)))
                # Remove duplicates while preserving order
                assets_to_try = list(dict.fromkeys(assets_to_try))
            else:
                assets_to_try = random.sample(assets_to_trade, len(assets_to_trade))
            
            # Try assets until we find one with good setup
            for asset in assets_to_try:
                print(f"Analyzing asset: {asset}")
                
                # Get asset information and check if it's open
                asset_name, asset_data = await client.get_available_asset(asset, force_open=True)
                
                if not asset_data or not asset_data[2]:
                    print(f"Asset {asset} is closed. Trying another one.")
                    continue
                    
                # MEJORA: Obtener más datos históricos para análisis más profundo
                end_from_time = time.time()
                # Get more historical data for better analysis
                candles = await client.get_candles(asset_name, end_from_time, 14400, 60)  # Get 4 hours of 1-minute candles
                
                if not candles or len(candles) < 30:
                    print("Not enough candle data. Trying another asset.")
                    continue
                    
                # Process candles if needed
                candles_data = process_candles(candles, 60) if not candles[0].get("open") else candles
                
                # MEJORA: Análisis de volatilidad para asegurar mercado adecuado
                volatility = analyze_volatility(candles_data)
                if volatility == "low":
                    print(f"Asset {asset} has too low volatility. Seeking better opportunities.")
                    continue
                
                # Perform enhanced Smart Money analysis
                direction, reason, confidence = smart_money_analysis(candles_data)
                
                if not direction:
                    print(f"No clear setup found for {asset}. Trying another asset.")
                    continue
                
                # MEJORA: Solo operar con señales de alta confianza
                if confidence < 0.65:
                    print(f"Signal confidence too low ({confidence:.2f}). Seeking stronger setup.")
                    continue
                
                print(f"Smart Money analysis: {direction.upper()} - {reason}")
                print(f"Signal confidence: {confidence:.2f}")
                
                # MEJORA: Adaptar duración en base a volatilidad y tipo de señal
                trade_duration = base_duration
                if volatility == "high":
                    trade_duration = int(base_duration * 0.8)  # Shorter duration in high volatility
                elif "structure" in reason.lower() or "order block" in reason.lower():
                    trade_duration = int(base_duration * 1.2)  # Longer for structural trades
                
                # Get trade amount from risk manager with confidence factor
                trade_amount = risk_manager.get_trade_amount(asset, confidence)
                trade_amount = max(int(trade_amount), 10)  # Ensure minimum trade amount
                print(f"Risk-adjusted trade amount: ${trade_amount}")
                
                # Execute the trade
                print(f"Placing {direction.upper()} trade on {asset_name} for ${trade_amount}, duration {trade_duration}s")
                status, buy_info = await client.buy(trade_amount, asset_name, direction, trade_duration)
                
                if not status:
                    print("Trade execution failed.")
                    continue
                    
                print(f"Trade placed successfully. ID: {buy_info['id']}")
                print(f"Entry price: {buy_info.get('openPrice')}")
                print(f"Waiting {trade_duration} seconds for trade to complete...")
                
                # Wait for the trade duration
                await asyncio.sleep(trade_duration + 5)  # Add 5 seconds buffer
                
                # Check the result
                win = await client.check_win(buy_info["id"])
                profit = client.get_profit()
                
                # Update risk manager with confidence data
                risk_manager.update("win" if win else "loss", profit, asset, confidence)
                
                if win:
                    win_count += 1
                    print(f"WIN! Profit: ${profit}")
                else:
                    loss_count += 1
                    print(f"LOSS! Loss: ${profit}")
                    
                # Display current statistics
                current_balance = await client.get_balance()
                print(f"\nCurrent balance: ${current_balance}")
                print(f"Wins: {win_count}, Losses: {loss_count}")
                print(f"Win rate: {(win_count/(win_count+loss_count))*100:.2f}%" if (win_count+loss_count) > 0 else "N/A")
                print(f"Balance change: ${current_balance - initial_balance}")
                
                # MEJORA: Espera variable basada en resultado y confianza
                if win:
                    # Shorter delay after a win with high confidence
                    delay = random.randint(15, 30) if confidence > 0.8 else random.randint(20, 45)
                else:
                    # Longer delay after a loss
                    delay = random.randint(45, 90)
                    
                print(f"Waiting {delay} seconds before next trade...")
                await asyncio.sleep(delay)
                
                # Successfully placed a trade, move to next iteration
                break
            else:
                # If we tried all assets and none worked
                print("No suitable setup found across all assets. Waiting before retrying...")
                await asyncio.sleep(180)  # Wait 3 minutes before next scan
    
    except Exception as e:
        print(f"Error during trading: {e}")
    
    finally:
        # Show final results
        final_balance = await client.get_balance()
        print("\n=== TRADING SESSION COMPLETE ===")
        print(f"Initial balance: ${initial_balance}")
        print(f"Final balance: ${final_balance}")
        print(f"Total profit/loss: ${final_balance - initial_balance}")
        print(f"Trades executed: {win_count + loss_count}/{max_trades}")
        
        if win_count + loss_count > 0:
            print(f"Win rate: {(win_count/(win_count+loss_count))*100:.2f}% ({win_count} wins, {loss_count} losses)")
        
        print(f"Maximum drawdown: {risk_manager.max_drawdown:.2f}%")
        print(f"Longest win streak: {risk_manager.max_win_streak}")
        print(f"Longest loss streak: {risk_manager.max_loss_streak}")
        
        # MEJORA: Mostrar estadísticas de rendimiento por activo
        if risk_manager.asset_performance:
            print("\nPerformance by Asset:")
            for asset, data in risk_manager.asset_performance.items():
                wins = data["wins"]
                losses = data["losses"]
                if wins + losses > 0:
                    win_rate = (wins / (wins + losses)) * 100
                    print(f"{asset}: {win_rate:.1f}% win rate ({wins}W/{losses}L), Profit: ${data['profit']:.2f}")
        
        await client.close()
        print("Connection closed.")

async def main():
    # This will now run the automated trading system by default
    await automated_trading()

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("Closing at program.")
    finally:
        loop.close()