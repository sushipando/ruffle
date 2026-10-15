package {
    import flash.display.Sprite;
    import flash.filters.BlurFilter;
    import flash.filters.GlowFilter;

    // A clip with a glow, holding an empty child with a blur. The empty child has no
    // bounds, so the glow is drawn around the square.
    public class Test extends Sprite {
        public function Test() {
            var parent:Sprite = new Sprite();
            parent.graphics.beginFill(0x3366CC);
            parent.graphics.drawRect(60, 60, 80, 80);
            parent.graphics.endFill();
            parent.filters = [new GlowFilter(0xFF0000, 1, 24, 24, 4, 1)];

            var empty:Sprite = new Sprite();
            empty.filters = [new BlurFilter(8, 8, 1)];
            parent.addChild(empty);

            addChild(parent);
        }
    }
}
