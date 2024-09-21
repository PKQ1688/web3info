static MY_STATIC: i32 = 42;

fn main() {
    const SECOND_HOUR: usize = 3600;
    const SECOND_DAY: usize = SECOND_HOUR * 24;

    println!("{} ", SECOND_DAY);

    {
        const SE: usize = 1_000;
        println!("{} ", SE);
    }

    println!("{} ", SECOND_HOUR);
    println!("{MY_STATIC} ");
}
