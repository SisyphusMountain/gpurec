// Shared simulation utilities

use rand::Rng;

/// Draws an exponential waiting time for the next event.
/// If rates are zero, it is possible that total_rate is zero, in which case we return infinity
/// to indicate no more events will occur.
/// We will therefore only get speciations, and this edge case must be handled
/// elsewhere throughout the code.
#[inline]
pub(crate) fn draw_waiting_time<R: Rng>(total_rate: f64, rng: &mut R) -> f64 {
    if total_rate > 0.0 {
        let u: f64 = rng.gen();
        -u.ln() / total_rate
    } else {
        f64::INFINITY
    }
}
