package {
    import flash.display.Sprite;
    import flash.text.TextField;
    import flash.text.TextFieldType;
    import flash.text.TextFormat;

    // Moving the caret of an input field without word wrap scrolls it sideways, so that
    // the caret is visible: to the nearest multiple of a quarter of the visible width.
    public class Test extends Sprite {
        public function Test() {
            var tf:TextField = new TextField();
            tf.type = TextFieldType.INPUT;
            tf.defaultTextFormat = new TextFormat("_sans", 12);
            tf.width = 100;
            tf.height = 20;
            addChild(tf);
            tf.text = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJ";
            trace("maxScrollH " + tf.maxScrollH);
            var xs:Array = [];
            for (var c:int = 0; c < tf.length; c++) xs.push(tf.getCharBoundaries(c).x);
            trace("char x " + xs.join(","));
            for each (var i:int in [0, 10, 15, 18, 20, 21, 25, 30, 40, 46, 45, 40, 35, 30, 25, 20, 10, 0]) {
                tf.setSelection(i, i);
                trace("setSelection " + i + " scrollH " + tf.scrollH);
            }

            tf.wordWrap = true;
            tf.setSelection(46, 46);
            trace("word wrap: setSelection 46 scrollH " + tf.scrollH);
        }
    }
}
