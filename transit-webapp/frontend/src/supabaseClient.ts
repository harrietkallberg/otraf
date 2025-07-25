import { createClient } from '@supabase/supabase-js'

const supabaseUrl = 'https://uaqfokbeqneynggspqwk.supabase.co'
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVhcWZva2JlcW5leW5nZ3NwcXdrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM0NTQwMjIsImV4cCI6MjA2OTAzMDAyMn0.m04qZFLCqgHZGCpFjWVGrX6mxskdlsARsul7MStgV_8'

export const supabase = createClient(supabaseUrl, supabaseKey)
