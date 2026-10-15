package {
    import flash.display.MovieClip;
    import flash.events.Event;
    import flash.events.UncaughtErrorEvent;

    public class Main extends MovieClip {
        public var body:MovieClip;
        private var ticks:int = 0;

        public function Main() {
            // The frame script errors are expected: don't show or print them.
            loaderInfo.uncaughtErrorEvents.addEventListener(UncaughtErrorEvent.UNCAUGHT_ERROR, function(e:UncaughtErrorEvent):void {
                e.preventDefault();
            });
            addEventListener(Event.ENTER_FRAME, function(e:Event):void {
                ticks++;
                if (ticks > 14) return;
                trace("tick " + ticks + ": body frame " + body.currentFrame);
                if (ticks == 4) {
                    trace("play()");
                    body.play();
                }
                if (ticks == 9) {
                    trace("gotoAndPlay(3)");
                    body.gotoAndPlay(3);
                }
            });
        }
    }
}
