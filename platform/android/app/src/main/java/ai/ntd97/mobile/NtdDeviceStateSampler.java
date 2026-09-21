package ai.ntd97.mobile;

import android.app.ActivityManager;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.BatteryManager;
import android.os.Build;
import android.os.PowerManager;

final class NtdDeviceStateSampler {
    static final class Snapshot {
        final long availableRamBytes;
        final int batteryPercent;
        final boolean charging;
        final int thermalState;
        final int latencyBudgetMs;

        Snapshot(
                long availableRamBytes,
                int batteryPercent,
                boolean charging,
                int thermalState,
                int latencyBudgetMs) {
            this.availableRamBytes = availableRamBytes;
            this.batteryPercent = batteryPercent;
            this.charging = charging;
            this.thermalState = thermalState;
            this.latencyBudgetMs = latencyBudgetMs;
        }
    }

    private NtdDeviceStateSampler() {}

    static Snapshot sample(Context context) {
        long availableRamBytes = 0L;
        ActivityManager activityManager = context.getSystemService(ActivityManager.class);
        if (activityManager != null) {
            ActivityManager.MemoryInfo memoryInfo = new ActivityManager.MemoryInfo();
            activityManager.getMemoryInfo(memoryInfo);
            availableRamBytes = Math.max(0L, memoryInfo.availMem);
        }

        int batteryPercent = 50;
        boolean charging = false;
        Intent battery = context.registerReceiver(
                null,
                new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
        if (battery != null) {
            int level = battery.getIntExtra(BatteryManager.EXTRA_LEVEL, -1);
            int scale = battery.getIntExtra(BatteryManager.EXTRA_SCALE, -1);
            if (level >= 0 && scale > 0) {
                batteryPercent = Math.max(0, Math.min(100, (level * 100) / scale));
            }

            int status = battery.getIntExtra(
                    BatteryManager.EXTRA_STATUS,
                    BatteryManager.BATTERY_STATUS_UNKNOWN);
            charging = status == BatteryManager.BATTERY_STATUS_CHARGING
                    || status == BatteryManager.BATTERY_STATUS_FULL;
        }

        int thermalState = 1;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            PowerManager powerManager = context.getSystemService(PowerManager.class);
            if (powerManager != null) {
                int androidThermal = powerManager.getCurrentThermalStatus();
                if (androidThermal >= PowerManager.THERMAL_STATUS_CRITICAL) {
                    thermalState = 4;
                } else if (androidThermal >= PowerManager.THERMAL_STATUS_SEVERE) {
                    thermalState = 3;
                } else if (androidThermal >= PowerManager.THERMAL_STATUS_MODERATE) {
                    thermalState = 2;
                }
            }
        }

        return new Snapshot(
                availableRamBytes,
                batteryPercent,
                charging,
                thermalState,
                100);
    }
}
