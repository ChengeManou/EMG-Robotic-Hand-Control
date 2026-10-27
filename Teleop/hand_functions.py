import initialise
import pinky, ring, middle, index, thumb
initialise.initialize_hand()

def open_hand():
    pinky.flex()
    ring.flex()
    middle.flex()
    index.flex()
    thumb.flex()
    
def close_hand():
    pinky.close()
    ring.close()
    middle.close()
    index.close()
    thumb.close()
    
def pinch():
    index.close()
    thumb.close()
    
    pinky.moveto(100)
    ring.moveto(100)
    middle.moveto(100)

def tripod():
    thumb.close()
    index.close()
    middle.close()

    ring.moveto(50)
    pinky.moveto(50)
    
def index_extension():
    index.flex()

    thumb.close()
    middle.close()
    ring.close()
    pinky.close()
    
def thumb_extension():
    thumb.flex()

    index.close()
    middle.close()
    ring.close()
    pinky.close()
    
def middle_extension():
    
    middle.flex()

    index.close()
    thumb.close()
    ring.close()
    pinky.close()
    
def ring_extension():
    ring.flex()
    thumb.close()
    index.close()
    middle.close()
    pinky.close()
 
 
def pinky_extension():
    pinky.flex()
    thumb.close()
    index.close()
    middle.close()
    ring.close()

def peace():
    pinky.close()
    thumb.close()
    index.flex()
    middle.flex()
    ring.close()

def rocknroll():
    pinky.flex()
    thumb.close()
    index.flex()
    middle.close()
    ring.close()

