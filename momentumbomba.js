// This Pine Script® code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © isfatihttgundz

//@version=5
indicator("Momentum Bomba", overlay=true)

// Calculate indicators
ema9 = ta.ema(close, 8)
ema21 = ta.ema(close, 21)
rsi = ta.rsi(close, 13)
vol = volume
volAvg = ta.sma(vol, 20)

// Check for new 20-bar high
isNew20BarHigh = close >= ta.highest(close, 13)[1]

// Buy condition
buySignal = ta.crossover(ema9, ema21) and close > ema21 and rsi > 55 and rsi < 80 and vol > volAvg * 2 
shotSignal = ta.crossover(ema9, ema21) and rsi < 60  
// Sell condition (EMA crossunder)
sellSignal = ta.crossunder(ema9, ema21)

// Plot buy/sell icons
plotshape(buySignal, "Buy", shape.triangleup, location.belowbar, color.new(color.green, 0), size=size.normal)
plotshape(sellSignal, "Sell", shape.triangledown, location.abovebar, color.new(color.red, 0), size=size.normal)
plotshape(shotSignal, "Shot", shape.triangleup, location.belowbar, color.new(#d4ea69, 0), size=size.normal)

// Optional: Plot EMAs for reference
plot(ema9, "EMA 9", color=color.new(color.blue, 70))
plot(ema21, "EMA 21", color=color.new(color.red, 70))
