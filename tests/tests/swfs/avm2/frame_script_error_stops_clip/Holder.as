package {
    import flash.display.MovieClip;
    import flash.display.SimpleButton;

    public class Holder extends MovieClip {
        public var btn:SimpleButton;

        public function Holder() {
            trace("Holder ctor start, parent.__id0_=" + (parent ? parent["__id0_"] : "no parent") + ", btn=" + btn);
            super();
            trace("Holder ctor end, parent.__id0_=" + (parent ? parent["__id0_"] : "no parent") + ", btn=" + btn);
        }
    }
}
