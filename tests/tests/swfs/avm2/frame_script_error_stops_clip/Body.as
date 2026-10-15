package {
    import flash.display.MovieClip;

    // Frame 2's script throws before it stops the clip: Flash stops the clip anyway.
    public class Body extends MovieClip {
        public var __id0_:MovieClip;

        public function Body() {
            trace("Body ctor");
            addFrameScript(1, frame2);
        }

        internal function frame2():* {
            trace("Body frame 2 script, __id0_=" + __id0_);
            __id0_.missing.prop = 1;
            stop();
            trace("Body stopped at " + currentFrame);
        }
    }
}
