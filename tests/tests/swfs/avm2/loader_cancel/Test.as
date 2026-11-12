package {
    import flash.display.*;
    import flash.events.*;
    import flash.net.*;
    import flash.system.*;

    // Starting another load with a Loader cancels the load in progress, and so does
    // Loader.close(): a cancelled load fires no events and defines no classes. A load
    // started from the ioError handler of a failed load replaces it.
    public class Test extends MovieClip {
        private var dom:ApplicationDomain = new ApplicationDomain(ApplicationDomain.currentDomain);
        private var ldr:Loader = new Loader();
        private var events:Array = [];
        private var frame:int = 0;

        public function Test() {
            loaderInfo.uncaughtErrorEvents.addEventListener(UncaughtErrorEvent.UNCAUGHT_ERROR, onError);
            for each (var type:String in [Event.INIT, Event.COMPLETE, IOErrorEvent.IO_ERROR]) {
                ldr.contentLoaderInfo.addEventListener(type, onEvent);
            }
            trace("load ItemA.swf, then ItemB.swf right away");
            ldr.load(new URLRequest("ItemA.swf"), new LoaderContext(false, dom));
            ldr.load(new URLRequest("ItemB.swf"), new LoaderContext(false, dom));
            addEventListener(Event.ENTER_FRAME, onFrame);
        }

        private function onError(e:UncaughtErrorEvent):void {
            trace("uncaught " + e.error);
            e.preventDefault();
        }

        private function onEvent(e:Event):void {
            events.push(e.type);
            if (e.type == Event.COMPLETE) {
                trace("complete " + ldr.contentLoaderInfo.url.split("/").pop() + " " + definitions());
            } else if (e.type == IOErrorEvent.IO_ERROR) {
                trace("ioError, loading ItemC.swf from the handler");
                ldr.load(new URLRequest("ItemC.swf"), new LoaderContext(false, dom));
            }
        }

        private function definitions():String {
            return "ItemA=" + dom.hasDefinition("ItemA") + " ItemB=" + dom.hasDefinition("ItemB") + " ItemC=" + dom.hasDefinition("ItemC");
        }

        private function onFrame(e:Event):void {
            frame++;
            if (frame == 5) {
                trace("events: " + events.join(" "));
                events = [];
                trace("load ItemC.swf, then close()");
                ldr.load(new URLRequest("ItemC.swf"), new LoaderContext(false, dom));
                ldr.close();
            } else if (frame == 10) {
                trace("events: " + events.join(" "));
                trace(definitions());
                events = [];
                trace("load missing.swf");
                ldr.load(new URLRequest("missing.swf"), new LoaderContext(false, dom));
            } else if (frame == 16) {
                trace("events: " + events.join(" "));
                removeEventListener(Event.ENTER_FRAME, onFrame);
            }
        }
    }
}
