package {
    import flash.display.MovieClip;

    // Frame 2 places `__id0_`, which contains a button with a movie clip in its up state.
    // Constructing that button runs frame 2's script before `__id0_` is set, so the script
    // throws before it stops the clip: Flash stops the clip anyway.
    public class Body extends MovieClip {
        public var __id0_:MovieClip;

        public function Body() {
            trace("Body ctor");
            addFrameScript(1, frame2);
        }

        internal function frame2():* {
            trace("Body frame 2 script, __id0_=" + __id0_);
            __id0_.label = "Next";
            stop();
            trace("Body stopped at " + currentFrame);
        }
    }
}
