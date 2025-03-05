import graphviz

# Create an activity diagram for the AI's Dynamic Move Selection feature
dot = graphviz.Digraph(format='png', engine='dot', graph_attr={'splines': 'polyline'})

# Define start and end points
dot.node('Start', 'Start', shape='ellipse', style='filled', fillcolor='lightgrey')
dot.node('End', 'End', shape='ellipse', style='filled', fillcolor='grey')

# Define key activity steps
dot.node('Analyze', 'Analyze Battle State', shape='box', style='filled', fillcolor='lightblue')
dot.node('Evaluate', 'Evaluate Available Moves', shape='box', style='filled', fillcolor='lightyellow')
dot.node('Decide', 'Decide Best Action', shape='diamond', style='filled', fillcolor='lightgreen')
dot.node('Attack', 'Perform Attack', shape='box', style='filled', fillcolor='lightcoral')
dot.node('Switch', 'Switch Pokémon', shape='box', style='filled', fillcolor='lightpink')
dot.node('Item', 'Use Item', shape='box', style='filled', fillcolor='lightcyan')
dot.node('Wait', 'Wait for Opponent Action', shape='box', style='filled', fillcolor='lightblue')

# Define transitions
dot.edge('Start', 'Analyze', label='Begin Turn')
dot.edge('Analyze', 'Evaluate', label='Check AI & Opponent Pokémon')
dot.edge('Evaluate', 'Decide', label='Assess Move Strength & Effectiveness')

# Decision branching
dot.edge('Decide', 'Attack', label='If attacking is best option')
dot.edge('Decide', 'Switch', label='If switching is best option')
dot.edge('Decide', 'Item', label='If using an item is best')

# After an action, wait for opponent
dot.edge('Attack', 'Wait', label='Execute Move')
dot.edge('Switch', 'Wait', label='Change Pokémon')
dot.edge('Item', 'Wait', label='Use Healing or Boost Item')

# Transition to end of turn
dot.edge('Wait', 'End', label='Opponent Completes Turn')

# Render and display the diagram
dot.render('pokemon_ai_activity_diagram', format='png', view=True)
