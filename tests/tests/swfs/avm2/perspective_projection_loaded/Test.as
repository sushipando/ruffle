package {
    import flash.display.*;
    import flash.events.Event;
    import flash.geom.PerspectiveProjection;
    import flash.net.URLRequest;

    [SWF(width="400", height="300")]
    public class Test extends Sprite {
        private var loader:Loader = new Loader();

        public function Test() {
            trace("// main root");
            printProps(root.transform.perspectiveProjection);

            loader.contentLoaderInfo.addEventListener(Event.COMPLETE, onComplete);
            loader.load(new URLRequest("child/child.swf"));
            addChild(loader);
        }

        private function onComplete(e:Event):void {
            var content:DisplayObject = loader.content;
            trace("// loaded root");
            printProps(content.transform.perspectiveProjection);

            var inner:Sprite = new Sprite();
            DisplayObjectContainer(content).addChild(inner);
            trace("// root of a child of the loaded root");
            trace(inner.root == content);
            printProps(inner.root.transform.perspectiveProjection);

            trace("// loaded root after removing the Loader");
            removeChild(loader);
            printProps(content.transform.perspectiveProjection);

            trace("// new Sprite");
            printProps(new Sprite().transform.perspectiveProjection);
        }

        private function printProps(pp:PerspectiveProjection):void {
            if (pp == null) {
                trace("null");
                return;
            }
            trace("fieldOfView = " + pp.fieldOfView);
            trace("focalLength = " + pp.focalLength);
            trace("projectionCenter = " + pp.projectionCenter);
        }
    }
}
