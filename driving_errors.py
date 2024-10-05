
driving_errors = [
    "The vehicle drifted into the next lane without signaling.",
    "The vehicle did not stop for a pedestrian crossing at a zebra crossing.",
    "The vehicle ignored a yield sign and proceeded without caution.",
    "The vehicle accelerated rapidly in a school zone where speeds should be reduced.",
    "The vehicle failed to give way to oncoming traffic at a roundabout.",
    "The vehicle attempted a left turn from the wrong lane, cutting across traffic.",
    "The vehicle did not come to a complete stop at a red light.",
    "The vehicle missed a turn and attempted to proceed.",
    "The vehicle turned right without checking for cyclists in the bike lane.",
    "The vehicle overran the stop line at a traffic light and blocked the crosswalk.",
    "The vehicle failed to signal before changing lanes on the highway.",
    "The vehicle drove too close to the curb, scraping the wheel against it.",
    "The vehicle sped through a yellow light instead of slowing down.",
    "The vehicle did not adjust speed in heavy rain and skidded at a corner.",
    "The vehicle took an illegal U-turn at a no U-turn sign.",
    "The vehicle rolled backward slightly while waiting on a hill, almost hitting the car behind.",
    "The vehicle merged onto the highway without checking for approaching traffic.",
    "The vehicle didn’t maintain a safe following distance, leading to a near rear-end collision.",
    "The vehicle veered off the road momentarily while navigating a sharp bend.",
    "The vehicle entered a one-way street from the wrong direction, causing confusion for oncoming vehicles."
]

# Manually labeled categories for each driving error
labels = [
    "Failed for Lane Position",  # for "The vehicle drifted into the next lane without signaling."
    "Failed to remain Stopped",  # for "The vehicle did not stop for a pedestrian crossing..."
    "Failed to follow Route",    # for "The vehicle ignored a yield sign..."
    "Failed to Slow",            # for "The vehicle accelerated rapidly in a school zone..."
    "Failed to follow Route",    # for "The vehicle failed to give way..."
    "Failed for Lane Position",  # for "The vehicle made a left turn from the wrong lane..."
    "Failed to remain Stopped",  # for "The vehicle did not come to a complete stop..."
    "Failed to follow Route",    # for "The vehicle missed a turn..."
    "Failed for Lane Position",  # for "The vehicle turned right without checking for cyclists..."
    "Failed to remain Stopped",  # for "The vehicle overran the stop line..."
    "Failed for Lane Position",  # for "The vehicle failed to signal before changing lanes..."
    "Failed for Lane Position",  # for "The vehicle drove too close to the curb..."
    "Failed to Slow",            # for "The vehicle sped through a yellow light..."
    "Failed to Slow",            # for "The vehicle did not adjust speed in heavy rain..."
    "Failed to follow Route",    # for "The vehicle took an illegal U-turn..."
    "Failed to remain Stopped",  # for "The vehicle rolled backward slightly..."
    "Failed for Lane Position",  # for "The vehicle merged onto the highway..."
    "Failed to maintain Speed",  # for "The vehicle didn’t maintain a safe following distance..."
    "Failed for Lane Position",  # for "The vehicle veered off the road..."
    "Failed to follow Route"     # for "The vehicle entered a one-way street..."
]