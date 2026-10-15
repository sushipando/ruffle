package {
    import flash.display.Sprite;
    import flash.events.Event;
    import flash.text.TextField;

    // `mouseEnabled = false` only affects the container itself: a text field inside it
    // can still be selected by dragging the mouse across it.
    public class Test extends Sprite {
        private var field:TextField = new TextField();
        private var frame:int = 0;

        public function Test() {
            var container:Sprite = new Sprite();
            container.mouseEnabled = false;
            addChild(container);

            field.selectable = true;
            field.width = 300;
            field.height = 30;
            field.x = 10;
            field.y = 10;
            field.text = "Some text to select by dragging the mouse";
            container.addChild(field);

            addEventListener(Event.ENTER_FRAME, function(e:Event):void {
                frame++;
                trace("frame " + frame + ": range selected " + (field.selectionEndIndex > field.selectionBeginIndex));
            });
        }
    }
}
